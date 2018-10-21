#ifndef __KFMDenseBlockSparseMatrix_OpenCL_H__
#define __KFMDenseBlockSparseMatrix_OpenCL_H__


#include <cstring>
#include <sstream>
#include <fstream>
#include <cstdlib>


//math
#include "KSquareMatrix.hh"
#include "KEMChunkedFileInterface.hh"
#include "KEMSparseMatrixFileInterface.hh"
#include "KFMDenseBlockSparseMatrixStructure.hh"
#include "KFMMessaging.hh"

//core (opencl)
#include "KOpenCLInterface.hh"
#include "KOpenCLKernelBuilder.hh"

#define DEBUG_VERBOSE



namespace KEMField
{

/**
*
*@file KDenseBlockSparseMatrix_OpenCL.hh
*@class KFMDenseBlockSparseMatrix_OpenCL
*@brief
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Oct  6 15:08:56 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/


template< typename ValueType>
class KFMDenseBlockSparseMatrix_OpenCL: public KSquareMatrix< ValueType >
{
    public:

        KFMDenseBlockSparseMatrix_OpenCL(std::string unique_id, unsigned int verbosity = 0):
            fUniqueID(unique_id),
            fVerbosity(verbosity)
        {
            fSuggestedMatrixElementBufferSize = 0;
            fSuggestedMatrixIndexBufferSize = 0;
            //now we construct the OpenCL kernel and buffers
            CollectDeviceProperties();
            ConstructKernel();
            fZero = 0.0;

            fRowSizes = new std::vector<unsigned int>();
            fColumnSizes  = new std::vector<unsigned int>();
            fNElements  = new std::vector<unsigned int>();
            fRowOffsets = new std::vector<unsigned int>();
            fColumnOffsets = new std::vector<unsigned int>();
            fElementOffsets = new std::vector<unsigned int>();

        };

        virtual ~KFMDenseBlockSparseMatrix_OpenCL()
        {
            delete fKernel;
            delete fRowBufferCL;
            delete fColumnBufferCL;
            delete fElementBufferCL;
            delete fInputVectorBufferCL;
            delete fOutputVectorBufferCL;

            delete fRowSizes;
            delete fColumnSizes;
            delete fNElements;
            delete fRowOffsets;
            delete fColumnOffsets;
            delete fElementOffsets;
        };

        virtual unsigned int Dimension() const {return (unsigned int)fDimension;};

        size_t GetSuggestedMatrixElementBufferSize() const {return fSuggestedMatrixElementBufferSize; };
        size_t GetSuggestedIndexBufferSize() const { return  fSuggestedMatrixIndexBufferSize;};
        size_t GetSuggestedMaximumRowWidth() const { return fNLocal;};

        virtual void Multiply(const KVector<ValueType>& x, KVector<ValueType>& y) const
        {
            //initialize input/output vectors
            for(size_t i=0; i<fDimension; i++)
            {
                fInputVector[i] = x(i);
                fOutputVector[i] = 0.0;
            }

            //copy the input vector x over to the GPU
            KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fInputVectorBufferCL, CL_TRUE, 0, fDimension*sizeof(CL_TYPE), &(fInputVector[0]) );

            //zero out the output vector on the GPU
            KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fOutputVectorBufferCL, CL_TRUE, 0, fDimension*sizeof(CL_TYPE), &(fOutputVector[0]) );

            if(fIsSingleBuffer)
            {
                SingleBufferMultiply();
            }
            else
            {
                MultipleBufferMultiply();
            }

            //copy the output vector back from the GPU
            KOpenCLInterface::GetInstance()->GetQueue().enqueueReadBuffer(*fOutputVectorBufferCL, CL_TRUE, 0, fDimension*sizeof(CL_TYPE), &(fOutputVector[0]) );

            //copy the output vector into y
            for(size_t i=0; i<fDimension; i++)
            {
                y[i] = fOutputVector[i];
            }
        }

        //following function must be defined but it is not implemented
        virtual const ValueType& operator()(unsigned int,unsigned int) const
        {
            return fZero;
        }


        void Initialize()
        {
            //construct file names from unique id

            //DBSMSF = dense block sparse matrix structure file
            fStructureFileName = KFMDenseBlockSparseMatrixStructure::StructureFilePrefix;
            fStructureFileName += fUniqueID;
            fStructureFileName += KFMDenseBlockSparseMatrixStructure::StructureFilePostfix;

            //DBSMRF = dense block sparse matrix row file
            fRowFileName = KFMDenseBlockSparseMatrixStructure::RowFilePrefix;
            fRowFileName += fUniqueID;
            fRowFileName += KFMDenseBlockSparseMatrixStructure::RowFilePostfix;

            //DBSMCF = dense block sparse matrix column file
            fColumnFileName = KFMDenseBlockSparseMatrixStructure::ColumnFilePrefix;
            fColumnFileName += fUniqueID;
            fColumnFileName += KFMDenseBlockSparseMatrixStructure::ColumnFilePostfix;

            //DBSMEF = dense block sparse matrix element file
            fElementFileName = KFMDenseBlockSparseMatrixStructure::ElementFilePrefix;
            fElementFileName += fUniqueID;
            fElementFileName += KFMDenseBlockSparseMatrixStructure::ElementFilePostfix;

            //read the structure file from disk
            bool result = false;
            KSAObjectInputNode< KFMDenseBlockSparseMatrixStructure >* structure_node;
            structure_node = new KSAObjectInputNode<KFMDenseBlockSparseMatrixStructure>( KSAClassName<KFMDenseBlockSparseMatrixStructure>::name() );
            KEMFileInterface::GetInstance()->ReadKSAFile(structure_node, fStructureFileName, result);

            if(result)
            {
                fMatrixStructure = *( structure_node->GetObject() );
                delete structure_node;
            }
            else
            {
                //error, abort
                delete structure_node;
                kfmout<<"KFMDenseBlockSparseMatrix_OpenCL::Initialize(): Error, structure file: "<<fStructureFileName<<" corrupt or not present."<<kfmendl;
                kfmexit(1);
            }

            if(fVerbosity > 2)
            {
                kfmout<<"Sparse matrix component has "<<fMatrixStructure.GetNBlocks()<<" blocks. "<<kfmendl;
                kfmout<<"Sparse matrix has "<<fMatrixStructure.GetNTotalNonZeroElements()<<" non-zero elements."<<kfmendl;

                double total_size = fMatrixStructure.GetNTotalNonZeroElements()*sizeof(double);
                total_size /= (1024.*1024);

                kfmout<<"Sparse matrix total size is "<<total_size<<" MB."<<kfmendl;

                double fraction = fMatrixStructure.GetNTotalNonZeroElements();
                fraction /= ((double)fMatrixStructure.GetDimension())*((double)fMatrixStructure.GetDimension());

                kfmout<<"Sparse matrix percentage of full system is: "<<fraction*100<<"%."<<kfmendl;
                kfmout<<"Sparse matrix component is divided across "<<fMatrixStructure.GetNBuffers()<<" buffers. "<<kfmendl;
            }

            fDimension = fMatrixStructure.GetDimension();

            fInputVector.resize(fDimension);
            fOutputVector.resize(fDimension);

            const std::vector<size_t>* rowSizes = fMatrixStructure.GetRowSizes();
            const std::vector<size_t>* columnSizes = fMatrixStructure.GetColumnSizes();
            const std::vector<size_t>* nElements = fMatrixStructure.GetNElements();
            const std::vector<size_t>* rowOffsets = fMatrixStructure.GetRowOffsets();
            const std::vector<size_t>* columnOffsets = fMatrixStructure.GetColumnOffsets();
            const std::vector<size_t>* elementOffsets = fMatrixStructure.GetMatrixElementOffsets();

            for(unsigned int i=0; i<rowSizes->size(); i++){fRowSizes->push_back( (unsigned int)rowSizes->at(i) ); };
            for(unsigned int i=0; i<columnSizes->size(); i++){fColumnSizes->push_back( (unsigned int)columnSizes->at(i) ); };
            for(unsigned int i=0; i<rowSizes->size(); i++){fNElements->push_back( (unsigned int)nElements->at(i) ); };
            for(unsigned int i=0; i<rowSizes->size(); i++){fRowOffsets->push_back( (unsigned int)rowOffsets->at(i) ); };
            for(unsigned int i=0; i<rowSizes->size(); i++){fColumnOffsets->push_back( (unsigned int)columnOffsets->at(i) ); };
            for(unsigned int i=0; i<rowSizes->size(); i++){fElementOffsets->push_back( (unsigned int)elementOffsets->at(i) ); };

            //prepare file interfaces
            fRowFileInterface = new KEMChunkedFileInterface();
            fColumnFileInterface = new KEMChunkedFileInterface();
            fElementFileInterface = new KEMChunkedFileInterface();

            //check that the row, column, and matrix element files exits
            bool row_exists = fRowFileInterface->DoesFileExist(fRowFileName);
            bool col_exists = fColumnFileInterface->DoesFileExist(fColumnFileName);
            bool elem_exists = fElementFileInterface->DoesFileExist(fElementFileName);

            if(!row_exists)
            {
                //abort, error
                delete fRowFileInterface;
                delete fColumnFileInterface;
                delete fElementFileInterface;
                kfmout<<"KFMDenseBlockSparseMatrix_OpenCL::Initialize(): Error, row file corrupt or not present."<<kfmendl;
                kfmout<<"Row file name = "<<fRowFileName<<kfmendl;
                kfmexit(1);
            }

            if(!col_exists)
            {
                //abort, error
                delete fRowFileInterface;
                delete fColumnFileInterface;
                delete fElementFileInterface;
                kfmout<<"KFMDenseBlockSparseMatrix_OpenCL::Initialize(): Error, column file corrupt or not present."<<kfmendl;
                kfmout<<"Column file name = "<<fColumnFileName<<kfmendl;
                kfmexit(1);
            }

            if(!elem_exists)
            {
                //abort, error
                delete fRowFileInterface;
                delete fColumnFileInterface;
                delete fElementFileInterface;
                kfmout<<"KFMDenseBlockSparseMatrix_OpenCL::Initialize(): Error, matrix element file corrupt or not present."<<kfmendl;
                kfmout<<"Element file name = "<<fElementFileName<<kfmendl;
                kfmexit(1);
            }

            //create the buffers needed to read the row, column and matrix elements
            if(fMatrixStructure.GetTotalNumberOfRowIndices() < fMatrixStructure.GetMaxIndexBufferSize() )
            {
                fRowIndices.resize(fMatrixStructure.GetTotalNumberOfRowIndices() + 1);
                fRowIndices_SizeType.resize(fMatrixStructure.GetTotalNumberOfRowIndices() + 1);
            }
            else
            {
                fRowIndices.resize(fMatrixStructure.GetMaxIndexBufferSize());
                fRowIndices_SizeType.resize(fMatrixStructure.GetMaxIndexBufferSize());
            }

            if(fMatrixStructure.GetTotalNumberOfColumnIndices() < fMatrixStructure.GetMaxIndexBufferSize() )
            {
                fColumnIndices.resize(fMatrixStructure.GetTotalNumberOfColumnIndices() + 1);
                fColumnIndices_SizeType.resize(fMatrixStructure.GetTotalNumberOfColumnIndices() + 1);
            }
            else
            {
                fColumnIndices.resize(fMatrixStructure.GetMaxIndexBufferSize());
                fColumnIndices_SizeType.resize(fMatrixStructure.GetMaxIndexBufferSize());
            }

            if(fMatrixStructure.GetNTotalNonZeroElements() < fMatrixStructure.GetMaxMatrixElementBufferSize() )
            {
                fMatrixElements.resize(fMatrixStructure.GetNTotalNonZeroElements() + 1);
            }
            else
            {
                fMatrixElements.resize(fMatrixStructure.GetMaxMatrixElementBufferSize());
            }

            fIsSingleBuffer = false;
            if(fMatrixStructure.GetNBuffers() == 1)
            {
                //only need a single buffer read
                //so we load the rows, columns, and matrix elements into memory
                //otherwise these will be buffered/read when a matrix-vector product is performed
                fIsSingleBuffer = true;

                fRowFileInterface->OpenFileForReading(fRowFileName);
                fRowFileInterface->Read(fMatrixStructure.GetBufferRowIndexSize(0), &(fRowIndices_SizeType[0]) );
                fRowFileInterface->CloseFile();

                //move size type into unsigned int
                for(unsigned int i=0; i<fRowIndices_SizeType.size(); i++)
                {
                    fRowIndices[i] = (unsigned int)fRowIndices_SizeType[i];
                }

                fColumnFileInterface->OpenFileForReading(fColumnFileName);
                fColumnFileInterface->Read(fMatrixStructure.GetBufferColumnIndexSize(0), &(fColumnIndices_SizeType[0]) );
                fColumnFileInterface->CloseFile();

                //move size type into unsigned int
                for(unsigned int i=0; i<fColumnIndices_SizeType.size(); i++)
                {
                    fColumnIndices[i] = (unsigned int)fRowIndices_SizeType[i];
                }

                fElementFileInterface->OpenFileForReading(fElementFileName);
                fElementFileInterface->Read(fMatrixStructure.GetBufferMatrixElementSize(0), &(fMatrixElements[0]) );
                fElementFileInterface->CloseFile();
            }

            //now we construct the OpenCL buffers
            //check that the size of the local threads is greater of equal to the maximum row width
            //otherwise we have an error
            if(fNLocal < fMatrixStructure.GetMaxRowWidth() )
            {
                //error, abort
                kfmout<<"KFMDenseBlockSparseMatrix_OpenCL::Initialize(): Error, maximum row width "<<fMatrixStructure.GetMaxRowWidth()<<" is greater than number of local threads: "<<fNLocal<<". Please reduce the maximum allowable row width parameter. "<<kfmendl;
                kfmexit(1);
            }

            BuildBuffers();
            AssignBuffers();
        }


    protected:

        void CollectDeviceProperties()
        {
            size_t max_buffer_size = KOpenCLInterface::GetInstance()->GetDevice().getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
            size_t number_of_doubles = max_buffer_size/sizeof(CL_TYPE);
            size_t number_of_uints = max_buffer_size/sizeof(size_t);

            size_t n_doubles = KEMFIELD_SPARSE_MATRIX_BUFFER_SIZE_MB;
            n_doubles *= 1024*1024/sizeof(double);

            if(number_of_doubles < n_doubles)
            {
                fSuggestedMatrixElementBufferSize = number_of_doubles;
            }
            else
            {
                fSuggestedMatrixElementBufferSize = n_doubles;
            }

            size_t n_uints = KEMFIELD_SPARSE_MATRIX_BUFFER_SIZE_MB;
            n_uints *= 1024*1024/sizeof(size_t);

            if(number_of_uints < n_uints)
            {
                fSuggestedMatrixIndexBufferSize = number_of_uints;
            }
            else
            {
                fSuggestedMatrixIndexBufferSize = n_uints;
            }
        }

        void ConstructKernel()
        {
            //Get name of kernel source file
            std::stringstream clFile;
            clFile << KOpenCLInterface::GetInstance()->GetKernelPath() << "/kEMField_DenseBlockSparseMatrixVectorProduct_kernel.cl";

            KOpenCLKernelBuilder k_builder;
            fKernel = k_builder.BuildKernel(clFile.str(), std::string("DenseBlockSparseMatrixVectorProduct") );

            //get n-local
            fNLocal = fKernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(KOpenCLInterface::GetInstance()->GetDevice());

            //make sure that fNLocal is a power of 2, if not, then we need to make it the largest power of 2 <= fNLocal
            unsigned int power_of_two = 1;
            while( power_of_two*2 <= fNLocal )
            {
                power_of_two *= 2;
            };
            fNLocal = power_of_two;

            if(fVerbosity >= 2)
            {
                kfmout<<"KFMDenseBlockSparseMatrix_OpenCL::ConstructKernel: Number of local threads is: "<<fNLocal<<kfmendl;
            }
        }

        void BuildBuffers()
        {

            //__global unsigned int NCols,
            //__global unsigned int* Rows,
            //__global unsigned int* Columns,
            //__global CL_TYPE* Elements,
            //__global CL_TYPE* in_vector,
            //__global CL_TYPE* out_vector,
            //__local CL_TYPE* scratch1,
            //__local CL_TYPE* scratch2

            size_t max_row_size = fMatrixStructure.GetLargestRowSize();
            size_t max_col_size = fMatrixStructure.GetLargestColumnSize();
            size_t max_block_size = fMatrixStructure.GetLargestBlockSize();

            //create the row index buffer
            fRowBufferCL
            = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, max_row_size*sizeof(unsigned int));

            //create the column index buffer
            fColumnBufferCL
            = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, max_col_size*sizeof(unsigned int));

            //create the matrix element buffer
            fElementBufferCL
            = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, max_block_size*sizeof(CL_TYPE));

            //input and output vectors
            fInputVectorBufferCL
            = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, fDimension*sizeof(CL_TYPE));

            fOutputVectorBufferCL
            = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_WRITE, fDimension*sizeof(CL_TYPE));
        }

        void AssignBuffers()
        {
            //__global unsigned int NCols,
            //__global unsigned int* Rows,
            //__global unsigned int* Columns,
            //__global CL_TYPE* Elements,
            //__global CL_TYPE* in_vector,
            //__global CL_TYPE* out_vector,
            //__local CL_TYPE* scratch1,
            //__local CL_TYPE* scratch2

            // Set arguments to kernel
            fKernel->setArg(0, 0);
            fKernel->setArg(1, 0);
            fKernel->setArg(2, *fRowBufferCL);
            fKernel->setArg(3, *fColumnBufferCL);
            fKernel->setArg(4, *fElementBufferCL);
            fKernel->setArg(5, *fInputVectorBufferCL);
            fKernel->setArg(6, *fOutputVectorBufferCL);
            fKernel->setArg(7, fNLocal*sizeof(CL_TYPE), NULL);
            fKernel->setArg(8, fNLocal*sizeof(CL_TYPE), NULL);
        }

        void SingleBufferMultiply() const
        {
            //must buffer rows, columns, and matrix elements from the disk
            size_t start_block_id;
            size_t n_blocks;
            size_t block_id;

            //figure out the index of the start block and number of blocks in this buffer in the matrix structure
            start_block_id = fMatrixStructure.GetBufferStartBlockID(0);
            n_blocks = fMatrixStructure.GetBufferNumberOfBlocks(0);

            //loop over all blocks in buffer, apply each as a matrix multiplication
            for(size_t n = 0; n < n_blocks; n++)
            {
                //retrieve block information
                block_id = start_block_id + n;
                unsigned int nrows = (*fRowSizes)[block_id];
                unsigned int ncols = (*fColumnSizes)[block_id];

                //we are forced to use unsigned int by OpenCL
                //but be aware for very large blocks this number might be out of the
                //range of unsigned int
                unsigned int n_elements = nrows*ncols;

                fKernel->setArg(0, nrows);
                fKernel->setArg(1, ncols);

                //copy the buffer's over to the GPU
                KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fRowBufferCL, CL_TRUE, 0, nrows*sizeof(unsigned int), &(fRowIndices[ (*fRowOffsets)[block_id] ]) );

                KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fColumnBufferCL, CL_TRUE, 0, ncols*sizeof(unsigned int),  &(fColumnIndices[ (*fColumnOffsets)[block_id] ]) );

                KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fElementBufferCL, CL_TRUE, 0, n_elements*sizeof(CL_TYPE),  &(fMatrixElements[ (*fElementOffsets)[block_id] ]) );

                //now figure out the global number of items we need to run the kernel on
                //this should be fNLocal times the number of rows we have
                unsigned int nLocal = fNLocal;
                unsigned int nGlobal = nrows*fNLocal;

                //run the kernel
                KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fKernel, cl::NullRange, cl::NDRange(nGlobal), cl::NDRange(nLocal));
            }
        }


        void MultipleBufferMultiply() const
        {
            //must buffer rows, columns, and matrix elements from the disk
            size_t start_block_id;
            size_t n_blocks;
            size_t block_id;

            fRowFileInterface->OpenFileForReading(fRowFileName);
            fColumnFileInterface->OpenFileForReading(fColumnFileName);
            fElementFileInterface->OpenFileForReading(fElementFileName);

            for(size_t buffer_id=0; buffer_id < fMatrixStructure.GetNBuffers(); buffer_id++)
            {
                //read this buffer's data from disk
                fRowFileInterface->Read(fMatrixStructure.GetBufferRowIndexSize(buffer_id), &(fRowIndices[0]) );
                fColumnFileInterface->Read(fMatrixStructure.GetBufferColumnIndexSize(buffer_id), &(fColumnIndices[0]) );
                fElementFileInterface->Read(fMatrixStructure.GetBufferMatrixElementSize(buffer_id), &(fMatrixElements[0]) );

                //figure out the index of the start block and number of blocks in this buffer in the matrix structure
                start_block_id = fMatrixStructure.GetBufferStartBlockID(buffer_id);
                n_blocks = fMatrixStructure.GetBufferNumberOfBlocks(buffer_id);

                //loop over all blocks in buffer, apply each as a matrix multiplication
                for(size_t n = 0; n < n_blocks; n++)
                {
                    //retrieve block information
                    block_id = start_block_id + n;
                    unsigned int nrows = (*fRowSizes)[block_id];
                    unsigned int ncols = (*fColumnSizes)[block_id];

                    //we are forced to use unsigned int by OpenCL
                    //but be aware for very large blocks this number might be out of the
                    //range of unsigned int
                    unsigned int n_elements = nrows*ncols;

                    fKernel->setArg(0, nrows);
                    fKernel->setArg(1, ncols);

                    //copy the buffer's over to the GPU
                    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fRowBufferCL, CL_TRUE, 0, nrows*sizeof(unsigned int), &(fRowIndices[ (*fRowOffsets)[block_id] ]) );

                    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fColumnBufferCL, CL_TRUE, 0, ncols*sizeof(unsigned int),  &(fColumnIndices[ (*fColumnOffsets)[block_id] ]) );

                    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fElementBufferCL, CL_TRUE, 0, n_elements*sizeof(CL_TYPE),  &(fMatrixElements[ (*fElementOffsets)[block_id] ]) );

                    //now figure out the global number of items we need to run the kernel on
                    //this should be fNLocal times the number of rows we have
                    unsigned int nLocal = fNLocal;
                    unsigned int nGlobal = nrows*fNLocal;

                    //run the kernel
                    KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fKernel, cl::NullRange, cl::NDRange(nGlobal), cl::NDRange(nLocal));
                }
            }

            fRowFileInterface->CloseFile();
            fColumnFileInterface->CloseFile();
            fElementFileInterface->CloseFile();
        }

        ////////////////////////////////////////////////////////////////////////
        //data /////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////

        std::string fUniqueID;
        std::string fStructureFileName;
        std::string fRowFileName;
        std::string fColumnFileName;
        std::string fElementFileName;

        size_t fDimension;
        size_t fSuggestedMatrixElementBufferSize;
        size_t fSuggestedMatrixIndexBufferSize;
        unsigned int fVerbosity;

        //buffer sizes, these are determined by the properties
        //of the OpenCL device we are using and/or by the compilation
        //parameter KEMFIELD_SPARSE_MATRIX_BUFFER_SIZE_MB


        ////////////////////////////////////////////////////////////////////////

        bool fIsSingleBuffer;

        KFMDenseBlockSparseMatrixStructure fMatrixStructure;

        mutable KEMChunkedFileInterface* fRowFileInterface;
        mutable KEMChunkedFileInterface* fColumnFileInterface;
        mutable KEMChunkedFileInterface* fElementFileInterface;

        //temp work space
        mutable std::vector<ValueType> fInputVector;
        mutable std::vector<ValueType> fOutputVector;
        mutable std::vector<ValueType> fNumberOfColumns;

        mutable std::vector<unsigned int>* fRowSizes;
        mutable std::vector<unsigned int>* fColumnSizes;
        mutable std::vector<unsigned int>* fNElements;
        mutable std::vector<unsigned int>* fRowOffsets;
        mutable std::vector<unsigned int>* fColumnOffsets;
        mutable std::vector<unsigned int>* fElementOffsets;

        mutable std::vector<size_t> fRowIndices_SizeType;
        mutable std::vector<size_t> fColumnIndices_SizeType;

        mutable std::vector<unsigned int> fRowIndices;
        mutable std::vector<unsigned int> fColumnIndices;

        mutable std::vector<CL_TYPE> fMatrixElements;

        ValueType fZero;

        //OpenCL data
        mutable cl::Kernel* fKernel;
        mutable unsigned int fNLocal;
        mutable unsigned int fNGlobal;

        mutable cl::Buffer* fRowBufferCL;
        mutable cl::Buffer* fColumnBufferCL;
        mutable cl::Buffer* fElementBufferCL;
        mutable cl::Buffer* fInputVectorBufferCL;
        mutable cl::Buffer* fOutputVectorBufferCL;
};


}

#endif /* __KFMDenseBlockSparseMatrix_OpenCL_H__ */
