#ifndef KFMElectrostaticSparseBoundaryIntegralShellMatrix_OpenCL_HH__
#define KFMElectrostaticSparseBoundaryIntegralShellMatrix_OpenCL_HH__

#include "KFMDenseBlockSparseMatrixStructureGenerator.hh"
#include "KFMDenseBlockSparseMatrixStructure.hh"
#include "KFMDenseBlockSparseMatrix.hh"

#include "KBoundaryIntegralMatrix.hh"
#include "KSquareMatrix.hh"

#include "KSurfaceContainer.hh"

#include "KFMElectrostaticBoundaryIntegrator.hh"
#include "KFMElectrostaticBoundaryIntegratorEngine_OpenCL.hh"
#include "KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL.hh"

#include "KOpenCLInterface.hh"
#include "KOpenCLSurfaceContainer.hh"
#include "KFMElectrostaticNode.hh"

//buffer size in number of doubles

namespace KEMField
{

/*
*
*@file KFMElectrostaticSparseBoundaryIntegralShellMatrix_OpenCL.hh
*@class KFMElectrostaticSparseBoundaryIntegralShellMatrix_OpenCL
*@brief responsible for evaluating the sparse 'near-field' component of the BEM matrix
* using a block compressed row storage format for better caching
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Jan 29 14:53:59 EST 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template< typename FastMultipoleIntegrator >
class KFMElectrostaticSparseBoundaryIntegralShellMatrix_OpenCL: KSquareMatrix<typename FastMultipoleIntegrator::Basis::ValueType>
{
    public:

        typedef typename FastMultipoleIntegrator::Basis::ValueType ValueType;
        typedef KSquareMatrix<ValueType> Matrix;
        typedef KVector<ValueType> Vector;

        KFMElectrostaticSparseBoundaryIntegralShellMatrix_OpenCL(KSurfaceContainer& c, FastMultipoleIntegrator& integrator):
            fFastMultipoleIntegrator(integrator),
            fDimension(integrator.Dimension()),
            fUniqueID(integrator.GetUniqueIDString()),
            fElementBufferSize(0),
            fIndexBufferSize(0),
            fMaxRowWidth(UINT_MAX), //do not break up blocks for excessive width!!
            fVerbosity(integrator.GetVerbosity())
            {
                fBlockComputeKernel = NULL;
                fBlockReduceKernel = NULL;

                fRowSizeBufferCL = NULL;
                fColSizeBufferCL = NULL;
                fRowBufferCL = NULL;
                fColumnBufferCL = NULL;
                fBlockBufferCL = NULL;
                fInputVectorBufferCL = NULL;
                fOutputVectorBufferCL = NULL;

                //build the opencl container
                fOCLSurfaceContainer = new KOpenCLSurfaceContainer(c);

                //first we build the kernel
                CollectDeviceProperties();

                //determine the opencl flags
                std::stringstream ss;
                ss << fOCLSurfaceContainer->GetOpenCLFlags();
                ss << " -DKEMFIELD_INTEGRATORFILE_CL=<" << "kEMField_ElectrostaticNumericBoundaryIntegrals.cl" <<">";
                ss << " -DKEMFIELD_OCLFASTRWG=" << KEMFIELD_OPENCL_FASTRWG; /* variable defined via cmake */
                fBuildFlags = ss.str();

                BuildBlockComputeKernel();
                BuildBlockReduceKernel();

                std::cout<<"creating the opencl surface container"<<std::endl;

                size_t min_workgroup_size = fNComputeLocal;
                if(fNReduceLocal < min_workgroup_size){min_workgroup_size = fNReduceLocal;};

                fOCLSurfaceContainer->SetMinimumWorkgroupSizeForKernels(min_workgroup_size);
                std::cout<<"done creating the opencl surface container"<<std::endl;

                std::cout<<"building the ocl container objects"<<std::endl;
                fOCLSurfaceContainer->BuildOpenCLObjects();
                std::cout<<"done building the ocl container objects"<<std::endl;

                fInputVector.resize(fDimension);
                fOutputVector.resize(fDimension);
                Initialize();
                fZero = 0.0;
            };

        virtual ~KFMElectrostaticSparseBoundaryIntegralShellMatrix_OpenCL()
        {
            delete fOCLSurfaceContainer;

            delete fBlockComputeKernel;
            delete fBlockReduceKernel;

            delete fRowSizeBufferCL;
            delete fColSizeBufferCL;
            delete fRowBufferCL;
            delete fColumnBufferCL;
            delete fBlockBufferCL;
            delete fInputVectorBufferCL;
            delete fOutputVectorBufferCL;
        };

        virtual void Multiply(const KVector<ValueType>& x, KVector<ValueType>& y) const
        {
            //initialize input/output vectors
            for(size_t i=0; i<fDimension; i++)
            {
                fInputVector[i] = x( fOCLSurfaceContainer->GetSortedIndexFromNormalIndex(i) );
                fOutputVector[i] = 0.0;
            }

            //copy the input vector x over to the GPU
            KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fInputVectorBufferCL, CL_TRUE, 0, fDimension*sizeof(CL_TYPE), &(fInputVector[0]) );

            //zero out the output vector on the GPU
            KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fOutputVectorBufferCL, CL_TRUE, 0, fDimension*sizeof(CL_TYPE), &(fOutputVector[0]) );

            //no buffering done
            BufferedMultiply();

            //copy the output vector back from the GPU
            KOpenCLInterface::GetInstance()->GetQueue().enqueueReadBuffer(*fOutputVectorBufferCL, CL_TRUE, 0, fDimension*sizeof(CL_TYPE), &(fOutputVector[0]) );

            //copy the output vector into y
            for(size_t i=0; i<fDimension; i++)
            {
                y[ fOCLSurfaceContainer->GetNormalIndexFromSortedIndex(i) ] = fOutputVector[i];
            }
        }

        virtual unsigned int Dimension() const {return (unsigned int)fDimension;};

        //following function must be defined but it is not implemented
        virtual const ValueType& operator()(unsigned int, unsigned int) const
        {
            return fZero;
        }

    protected:


        void Initialize()
        {
            //next we determine the sparse matrix structure
            KFMDenseBlockSparseMatrixStructureGenerator<KFMElectrostaticNodeObjects, KSquareMatrix<ValueType> > dbsmGenerator;
            dbsmGenerator.SetDimension(fDimension);
            dbsmGenerator.SetUniqueID(fUniqueID);
            dbsmGenerator.SetMaxAllowableRowWidth(fMaxRowWidth);
            dbsmGenerator.SetMaxMatrixElementBufferSize(fElementBufferSize);
            dbsmGenerator.SetMaxIndexBufferSize(fIndexBufferSize);
            dbsmGenerator.SetVerbosity(fVerbosity);
            dbsmGenerator.Initialize();

            fFastMultipoleIntegrator.GetTree()->ApplyCorecursiveAction(&dbsmGenerator);

            dbsmGenerator.Finalize();

            const std::vector<size_t>* row_indices = dbsmGenerator.GetRowIndices();
            const std::vector<size_t>* col_indices = dbsmGenerator.GetColumnIndices();

            fStructure = *( dbsmGenerator.GetMatrixStructure() );
            fRowSizes = fStructure.GetRowSizes();
            fColumnSizes = fStructure.GetColumnSizes();

            //re-map column element indices
            size_t col_size = col_indices->size();
            fColumnIndices.clear();
            fColumnIndices.reserve(col_size);
            for(size_t i=0; i<col_size; i++)
            {
                fColumnIndices.push_back( fOCLSurfaceContainer->GetSortedIndexFromNormalIndex( (*col_indices)[i] ) );
            }

            //re-map row element indices
            size_t row_size = row_indices->size();
            fRowIndices.clear();
            fRowIndices.reserve(col_size);
            for(size_t i=0; i<row_size; i++)
            {
                fRowIndices.push_back( fOCLSurfaceContainer->GetSortedIndexFromNormalIndex( (*row_indices)[i] ) );
            }

            //now we build and assign and buffers
            BuildBuffers();
            AssignBuffers();
        }

////////////////////////////////////////////////////////////////////////////////

        void CollectDeviceProperties()
        {
            size_t max_buffer_size = KOpenCLInterface::GetInstance()->GetDevice().getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
            size_t number_of_doubles = max_buffer_size/sizeof(CL_TYPE);
            size_t number_of_uints = max_buffer_size/sizeof(size_t);


            size_t n_doubles = KEMFIELD_SPARSE_MATRIX_BUFFER_SIZE_MB;
            n_doubles *= 1024*1024/sizeof(double);

            if(number_of_doubles < n_doubles)
            {
                fElementBufferSize = number_of_doubles;
            }
            else
            {
                fElementBufferSize  = n_doubles;
            }

            size_t n_uints = KEMFIELD_SPARSE_MATRIX_BUFFER_SIZE_MB;
            n_uints *= 1024*1024/sizeof(size_t);

            if(number_of_uints < n_uints)
            {
                fIndexBufferSize = number_of_uints;
            }
            else
            {
                fIndexBufferSize = n_uints;
            }
        }

////////////////////////////////////////////////////////////////////////////////

        void BuildBlockComputeKernel()
        {
            //Get name of kernel source file
            std::stringstream clFile;
            clFile << KOpenCLInterface::GetInstance()->GetKernelPath();
            clFile << "/kEMField_KFMElectrostaticSparseShellMatrixVectorProduct_kernel.cl";

            KOpenCLKernelBuilder k_builder;
            fBlockComputeKernel = k_builder.BuildKernel(clFile.str(),
                                std::string("ElectrostaticSparseShellMatrixVectorProduct_ComputeBlock"),
                                fBuildFlags);

            //get n-local
            fNComputeLocal = fBlockComputeKernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(KOpenCLInterface::GetInstance()->GetDevice());


            if(fVerbosity > 2)
            {
                kfmout<<"KFMElectrostaticSparseBoundaryIntegralShellMatrix_OpenCL::BuildBlockComputeKernel: Number of local threads is: "<<fNComputeLocal<<kfmendl;
            }
        }

        void BuildBlockReduceKernel()
        {
            //Get name of kernel source file
            std::stringstream clFile;
            clFile << KOpenCLInterface::GetInstance()->GetKernelPath();
            clFile << "/kEMField_KFMElectrostaticSparseShellMatrixVectorProduct_kernel.cl";

            KOpenCLKernelBuilder k_builder;
            fBlockReduceKernel = k_builder.BuildKernel(clFile.str(),
                                std::string("ElectrostaticSparseShellMatrixVectorProduct_ReduceBlock"),
                                fBuildFlags);

            //get n-local
            fNReduceLocal = fBlockReduceKernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(KOpenCLInterface::GetInstance()->GetDevice());

            if(fVerbosity > 2)
            {
                kfmout<<"KFMElectrostaticSparseBoundaryIntegralShellMatrix_OpenCL::BuildBlockReduceKernel: Number of local threads is: "<<fNReduceLocal<<kfmendl;
            }
        }


        void BuildBuffers()
        {
            size_t max_blocks = fStructure.GetMaxNumberOfBlocksInAnyBuffer();

            std::cout<<"max n blocks in buffer = "<<max_blocks<<std::endl;

            //create the row index buffer
            fRowSizeBufferCL
            = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, max_blocks*sizeof(size_t));

            //create the column index buffer
            fColSizeBufferCL
            = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, max_blocks*sizeof(size_t));

            //create the row index buffer
            size_t max_row_buffer_size = fStructure.GetMaxNumberOfRowIndicesInAnyBuffer();

            std::cout<<"max row buffer size = "<<max_row_buffer_size<<std::endl;

            fRowBufferCL
            = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, max_row_buffer_size*sizeof(size_t));

            //create the column index buffer
            size_t max_col_buffer_size = fStructure.GetMaxNumberOfColumnIndicesInAnyBuffer();

            std::cout<<"max_col_buffer_size = "<<max_col_buffer_size<<std::endl;

            fColumnBufferCL
            = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, max_col_buffer_size*sizeof(size_t));


            std::cout<<"element buffer size = "<<fElementBufferSize<<std::endl;
            std::cout<<"max actual buffer size = "<<fStructure.GetMaxNumberOfElementsInAnyBuffer()<<std::endl;

            fBlockBufferCL
            = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_WRITE, fElementBufferSize*sizeof(CL_TYPE));

            //input and output vectors
            fInputVectorBufferCL
            = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, fDimension*sizeof(CL_TYPE));

            fOutputVectorBufferCL
            = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_WRITE, fDimension*sizeof(CL_TYPE));
        }

        void AssignBuffers()
        {
            //__global const short* shapeInfo, //fixed argument
            //__global const CL_TYPE* shapeData, //fixed argument
            //__global const int* boundaryInfo, //fixed argument
            //__global const CL_TYPE* boundaryData, //fixed argument
            //const size_t TotalNElements,
            //const size_t TotalNBlocks,
            //__global size_t* RowSizes,
            //__global size_t* ColSizes,
            //__global size_t* Rows,
            //__global size_t* Columns,
            //__global CL_TYPE* in_vector,
            //__global CL_TYPE* block_data

            fBlockComputeKernel->setArg(0, *(fOCLSurfaceContainer->GetShapeInfo()) );
            fBlockComputeKernel->setArg(1,*(fOCLSurfaceContainer->GetShapeData()) );
            fBlockComputeKernel->setArg(2, *(fOCLSurfaceContainer->GetBoundaryInfo()) );
            fBlockComputeKernel->setArg(3, *(fOCLSurfaceContainer->GetBoundaryData()) );

            fBlockComputeKernel->setArg(4, 0); //total number of non-zero elements in buffer
            fBlockComputeKernel->setArg(5, 0); //total number of blocks
            fBlockComputeKernel->setArg(6, *fRowSizeBufferCL); //number of rows for each block
            fBlockComputeKernel->setArg(7, *fColSizeBufferCL); //number of columns for each block
            fBlockComputeKernel->setArg(8, *fRowBufferCL); //row indexes
            fBlockComputeKernel->setArg(9, *fColumnBufferCL); //column indexes
            fBlockComputeKernel->setArg(10, *fInputVectorBufferCL);
            fBlockComputeKernel->setArg(11, *fBlockBufferCL);


            //const size_t TotalNBlocks,
            //__global size_t* RowSizes,
            //__global size_t* ColSizes,
            //__global size_t* Rows,
            //__global CL_TYPE* block_data,
            //__global CL_TYPE* out_vector

            fBlockReduceKernel->setArg(0, 0); //total number of blocks
            fBlockReduceKernel->setArg(1, *fRowSizeBufferCL);
            fBlockReduceKernel->setArg(2, *fColSizeBufferCL);
            fBlockReduceKernel->setArg(3, *fRowBufferCL);
            fBlockReduceKernel->setArg(4, *fBlockBufferCL);
            fBlockReduceKernel->setArg(5, *fOutputVectorBufferCL);
        }

        void BufferedMultiply() const
        {
            size_t row_start = 0;
            size_t col_start = 0;

            for(size_t buffer_id=0; buffer_id < fStructure.GetNBuffers(); buffer_id++)
            {
                size_t start_block_id = fStructure.GetBufferStartBlockID(buffer_id);
                size_t n_blocks = fStructure.GetBufferNumberOfBlocks(buffer_id);

                fRowSizeBuffer.clear();
                fColSizeBuffer.clear();

                size_t n_elem = 0;
                size_t n_rows = 0;
                size_t n_cols = 0;

                for(size_t n = 0; n < n_blocks; n++)
                {
                    //retrieve block information
                    size_t block_id = start_block_id + n;
                    size_t row_size = (*fRowSizes)[block_id];
                    size_t col_size = (*fColumnSizes)[block_id];
                    n_elem += row_size*col_size;
                    n_rows += row_size;
                    n_cols += col_size;

                    fRowSizeBuffer.push_back(row_size);
                    fColSizeBuffer.push_back(col_size);
                }


////////////////////////////////////////////////////////////////////////////////

                fBlockComputeKernel->setArg(4, n_elem);
                fBlockComputeKernel->setArg(5, n_blocks);

                //copy the buffer's over to the GPU
                KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fRowSizeBufferCL, CL_TRUE, 0, n_blocks*sizeof(size_t), &(fRowSizeBuffer[0]) );

                KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fColSizeBufferCL, CL_TRUE, 0, n_blocks*sizeof(size_t), &(fColSizeBuffer[0]) );

                KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fRowBufferCL, CL_TRUE, 0, n_rows*sizeof(size_t), &(fRowIndices[row_start]) );

                KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fColumnBufferCL, CL_TRUE, 0, n_cols*sizeof(size_t), &(fColumnIndices[col_start]) );

                //now figure out the global number of items we need to run the block compute kernel on
                size_t nDummy = fNComputeLocal - (n_elem%fNComputeLocal);
                if(nDummy == fNComputeLocal){nDummy = 0;};

                //run the kernel

//                cl::Event block_compute_event;
                KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fBlockComputeKernel, cl::NullRange,  cl::NDRange(n_elem + nDummy), cl::NDRange(fNComputeLocal), NULL, NULL);
//                block_compute_event.wait();

                fBlockReduceKernel->setArg(0, n_blocks);

                //figure out the global number of items to run the reduce kernel on
                nDummy = fNReduceLocal - (n_blocks%fNReduceLocal);
                if(nDummy == fNReduceLocal){nDummy = 0;};

                //run the kernel
                KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fBlockReduceKernel, cl::NullRange,  cl::NDRange(n_blocks + nDummy), cl::NDRange(fNReduceLocal));

                row_start += n_rows;
                col_start += n_cols;
            }
        }

////////////////////////////////////////////////////////////////////////////////

        //data
        FastMultipoleIntegrator& fFastMultipoleIntegrator;
        KOpenCLSurfaceContainer* fOCLSurfaceContainer;
        size_t fDimension;
        std::string fUniqueID;
        size_t fElementBufferSize;
        size_t fIndexBufferSize;
        size_t fMaxRowWidth;
        size_t fVerbosity;
        ValueType fZero;

        //matrix structure
        KFMDenseBlockSparseMatrixStructure fStructure;
        const std::vector<size_t>* fRowSizes;
        const std::vector<size_t>* fColumnSizes;

        std::vector<size_t> fRowIndices; //re-mapped to the opencl indices!
        std::vector<size_t> fColumnIndices; //this is re-mapped to the opencl indices

        //OpenCL data
        std::string fBuildFlags;

        mutable cl::Kernel* fBlockComputeKernel;
        mutable cl::Kernel* fBlockReduceKernel;

        mutable size_t fNComputeLocal;
        mutable size_t fNReduceLocal;

        mutable std::vector<ValueType> fInputVector;
        mutable std::vector<ValueType> fOutputVector;

        //buffer for the row/column sizes of each block
        mutable std::vector<size_t > fRowSizeBuffer;
        mutable std::vector<size_t > fColSizeBuffer;
        mutable cl::Buffer* fRowSizeBufferCL;
        mutable cl::Buffer* fColSizeBufferCL;

        //buffers for the row/column indices
        mutable cl::Buffer* fRowBufferCL;
        mutable cl::Buffer* fColumnBufferCL;

        //buffer for the matrix element manipulation
        mutable cl::Buffer* fBlockBufferCL;

        //input and output vectors
        mutable cl::Buffer* fInputVectorBufferCL;
        mutable cl::Buffer* fOutputVectorBufferCL;


};

}//end of KEMField namespace

#endif /* KFMElectrostaticSparseBoundaryIntegralShellMatrix_OpenCL_H__ */
