#ifndef KFMBatchedMultidimensionalFastFourierTransform_OpenCL_HH__
#define KFMBatchedMultidimensionalFastFourierTransform_OpenCL_HH__

#include <cstring>
#include <sstream>
#include <fstream>
#include <cstdlib>

//core (opencl)
#include "KOpenCLInterface.hh"

#include "KFMArrayWrapper.hh"
#include "KFMFastFourierTransformUtilities.hh"

namespace KEMField
{

/*
*
*@file KFMBatchedMultidimensionalFastFourierTransform_OpenCL.hh
*@class KFMBatchedMultidimensionalFastFourierTransform_OpenCL
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Nov 27 00:01:00 EST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<unsigned int NDIM>
class KFMBatchedMultidimensionalFastFourierTransform_OpenCL: public KFMUnaryArrayOperator< std::complex<double>, NDIM+1 >
{
    public:

        KFMBatchedMultidimensionalFastFourierTransform_OpenCL():KFMUnaryArrayOperator< std::complex<double>, NDIM+1 >(),
        fIsValid(false),
        fIsForward(true),
        fInitialized(false),
        fAllSpatialDimensionsAreEqual(true),
        fFillFromHostData(true),
        fReadOutDataToHost(true),
        fMaxBufferSize(0),
        fTotalDataSize(0),
        fOpenCLFlags(""),
        fFFTKernel(NULL),
        fSpatialDimensionBufferCL(NULL),
        fTwiddleBufferCL(NULL),
        fConjugateTwiddleBufferCL(NULL),
        fScaleBufferCL(NULL),
        fCirculantBufferCL(NULL),
        fDataBufferCL(NULL),
        fPermuationArrayCL(NULL),
        fNLocal(0){;};

        virtual ~KFMBatchedMultidimensionalFastFourierTransform_OpenCL()
        {
            delete fFFTKernel;
            delete fSpatialDimensionBufferCL;
            delete fTwiddleBufferCL;
            delete fConjugateTwiddleBufferCL;
            delete fScaleBufferCL;
            delete fCirculantBufferCL;
            delete fDataBufferCL;
            delete fPermuationArrayCL;
        };

        //control direction of FFT
        void SetForward(){fIsForward = true;}
        void SetBackward(){fIsForward = false;};

        //control whether GPU buffer is initialized from host data (fInput)
        void SetWriteOutHostDataTrue(){fFillFromHostData = true;};
        void SetWriteOutHostDataFalse(){fFillFromHostData = false;};

        //control whether result is read back to host (fOutput)
        void SetReadOutDataToHostTrue(){fReadOutDataToHost = true;};
        void SetReadOutDataToHostFalse(){fReadOutDataToHost = false;};

////////////////////////////////////////////////////////////////////////////////

        //opencl build flags used
        std::string GetOpenCLFlags() const {return fOpenCLFlags;};

////////////////////////////////////////////////////////////////////////////////

        //raw access to the OpenCL data buffer...use carefully
        cl::Buffer* GetDataBuffer(){return fDataBufferCL;};

////////////////////////////////////////////////////////////////////////////////

        virtual void Initialize()
        {
            //input and output must be set before being called

            if(!fInitialized) //can only be initialized once!
            {
                if(DoInputOutputDimensionsMatch())
                {
                    fIsValid = true;
                    this->fInput->GetArrayDimensions(fDimensionSize);
                    for(unsigned int i=0; i<NDIM; i++)
                    {
                        fSpatialDim[i] = fDimensionSize[i+1];
                    }
                }
                else
                {
                    fIsValid = false;
                    fInitialized = false;
                }

                if(fIsValid)
                {
                    ConstructWorkspace();
                    ConstructOpenCLKernels();
                    BuildBuffers();
                    AssignBuffers();
                    fInitialized = true;
                }
            }
        }

////////////////////////////////////////////////////////////////////////////////

        virtual void ExecuteOperation()
        {
            if(fIsValid && fInitialized)
            {
                //set the basic arguments
                unsigned int n_multdim_ffts = fDimensionSize[0];
                fFFTKernel->setArg(0, n_multdim_ffts); //number of complete multidimensional fft's to perform
                if(fIsForward)
                {
                    fFFTKernel->setArg(2, 1); //direction of FFT is forward
                }
                else
                {
                    fFFTKernel->setArg(2, 0); //direction of FFT is backward (inverse)
                }
                //arg 3 is dimensions, already written to GPU

                //write the data to the buffer if necessary
                FillDataBuffer();

                for(unsigned int D=0; D<NDIM; D++)
                {
                    //compute number of 1d fft's needed (n-global)
                    unsigned int n_global = fDimensionSize[0];
                    unsigned int n_local_1d_transforms = 1;
                    for(unsigned int i = 0; i < NDIM; i++)
                    {
                         if(i != D)
                         {
                            n_global *= fSpatialDim[i];
                            n_local_1d_transforms *= fSpatialDim[i];
                         };
                    };

                    //pad out n-global to be a multiple of the n-local
                    unsigned int nDummy = fNLocal - (n_global%fNLocal);
                    if(nDummy == fNLocal){nDummy = 0;};
                    n_global += nDummy;

                    cl::NDRange global(n_global);
                    cl::NDRange local(fNLocal);

                    fFFTKernel->setArg(1, D); //(index of the selected dimension) updated at each stage

                    if(fAllSpatialDimensionsAreEqual)
                    {
                        //no need to write the constants, as they are already on the gpu
                        //just fire up the kernel
                        KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fFFTKernel, cl::NullRange, global, local);
                    }
                    else
                    {
                        //we enqueue write the needed constants for this dimension
                        KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fTwiddleBufferCL, CL_TRUE, 0, fMaxBufferSize*sizeof(CL_TYPE2), &(fTwiddle[D][0]) );
                        KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fConjugateTwiddleBufferCL, CL_TRUE, 0, fMaxBufferSize*sizeof(CL_TYPE2), &(fConjugateTwiddle[D][0]) );
                        KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fScaleBufferCL, CL_TRUE, 0, fMaxBufferSize*sizeof(CL_TYPE2), &(fScale[D][0]) );
                        KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fCirculantBufferCL, CL_TRUE, 0, fMaxBufferSize*sizeof(CL_TYPE2), &(fCirculant[D][0]) );
                        KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fPermuationArrayCL, CL_TRUE, 0, fMaxBufferSize*sizeof(unsigned int), &(fPermuationArray[D][0]) );

                        //now enqueue the kernel
                        KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fFFTKernel, cl::NullRange, global, local);
                    }
                }

                //read the data from the buffer if necessary
                ReadOutDataBuffer();
            }
            else
            {
                std::cout<<"KFMBatchedMultidimensionalFastFourierTransform_OpenCL::ExecuteOperation: Not valid and initialized. Aborting."<<std::endl;
            }
        }

////////////////////////////////////////////////////////////////////////////////

    private:

        void ConstructWorkspace()
        {
            //figure out the size of all the data
            fTotalDataSize = KFMArrayMath::TotalArraySize<NDIM+1>(fDimensionSize);

            //figure out the size of the private buffers needed
            fMaxBufferSize = 0;
            fAllSpatialDimensionsAreEqual = true;
            unsigned int previous_dim = fSpatialDim[0];
            for(unsigned int i=0; i<NDIM; i++)
            {
                if(previous_dim != fSpatialDim[i]){fAllSpatialDimensionsAreEqual = false;};

                if(fSpatialDim[i] > fMaxBufferSize)
                {
                    fMaxBufferSize = fSpatialDim[i];
                }

                if( !(KFMBitReversalPermutation::IsPowerOfTwo(fSpatialDim[i])) )
                {
                    if(KFMFastFourierTransformUtilities::ComputeBluesteinArraySize(fSpatialDim[i]) > fMaxBufferSize)
                    {
                        fMaxBufferSize = KFMFastFourierTransformUtilities::ComputeBluesteinArraySize(fSpatialDim[i]);
                    }
                }
            }

            //create the build flags
            std::stringstream ss;
            ss << " -D FFT_NDIM=" << NDIM;
            ss << " -D FFT_BUFFERSIZE=" << fMaxBufferSize;
            ss << " -I " <<KOpenCLInterface::GetInstance()->GetKernelPath();
            fOpenCLFlags = ss.str();

            //now we need to compute all the auxiliary variables we need for the FFT
            for(unsigned int i=0; i<NDIM; i++)
            {
                unsigned int N = fSpatialDim[i];
                unsigned int M = N;

                if( !(KFMBitReversalPermutation::IsPowerOfTwo(N)) )
                {
                    M = KFMFastFourierTransformUtilities::ComputeBluesteinArraySize(N);
                }

                //resize to appropriate lengths
                fTwiddle[i].resize(M);
                fConjugateTwiddle[i].resize(M);
                fScale[i].resize(M);
                fCirculant[i].resize(M);
                fPermuationArray[i].resize(M);

                //compute the twiddle factors for this dimension
                KFMFastFourierTransformUtilities::ComputeTwiddleFactors(M, &(fTwiddle[i][0]) );

                //compute the conjugate twiddle factors
                KFMFastFourierTransformUtilities::ComputeConjugateTwiddleFactors(M, &(fConjugateTwiddle[i][0]) );

                //compute the bluestein scale factors for this dimension
                KFMFastFourierTransformUtilities::ComputeBluesteinScaleFactors(N, &(fScale[i][0]) );
                //compute the circulant vector for this dimension
                KFMFastFourierTransformUtilities::ComputeBluesteinCirculantVector(N, M, &(fTwiddle[i][0]), &(fScale[i][0]), &(fCirculant[i][0]) );

                KFMBitReversalPermutation::ComputeBitReversedIndicesBaseTwo(M, &(fPermuationArray[i][0]) );

            }

        }

////////////////////////////////////////////////////////////////////////////////

        void ConstructOpenCLKernels()
        {
            //Get name of kernel source file
            std::stringstream clFile;
            clFile << KOpenCLInterface::GetInstance()->GetKernelPath() << "/kEMField_KFMMultidimensionalFastFourierTransform_kernel.cl";

            //read in the source code
            std::string sourceCode;
            std::ifstream sourceFile(clFile.str().c_str());
            sourceCode = std::string(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));

            //Make program of the source code in the context
            cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));
            cl::Program program(KOpenCLInterface::GetInstance()->GetContext(), source);

            //set the build options
            std::stringstream options;
            options << GetOpenCLFlags();

            // Build program for these specific devices
            try
            {
                // use only target device!
                cl::vector<cl::Device> devices;
                devices.push_back( KOpenCLInterface::GetInstance()->GetDevice() );
                program.build(devices,options.str().c_str());
            }
            catch (cl::Error error)
            {
                std::cout<<__FILE__<<":"<<__LINE__<<std::endl;
                std::stringstream s;
                s<<"There was an error compiling the kernels.  Here is the information from the OpenCL C++ API:"<<std::endl;
                s<<error.what()<<"("<<error.err()<<")"<<std::endl;
                s<<"Build Status: "<<program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(KOpenCLInterface::GetInstance()->GetDevice())<<std::endl;
                s<<"Build Options:\t"<<program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(KOpenCLInterface::GetInstance()->GetDevice())<<std::endl;
                s<<"Build Log:\t "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(KOpenCLInterface::GetInstance()->GetDevice())<<std::endl;
                std::cout<<s.str()<<std::endl;
                std::exit(1);
            }

            // Make kernel
	        cl_int err_code = -1;
            try
            {
                fFFTKernel = new cl::Kernel(program, "MultidimensionalFastFourierTransform_Stage", &err_code);
            }
            catch(cl::Error error)
            {
                std::cout<<"Kernel construction failed with error code: "<<err_code<<std::endl;
                std::exit(1);
            }

            //get n-local
            fNLocal = fFFTKernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(KOpenCLInterface::GetInstance()->GetDevice());
        }

////////////////////////////////////////////////////////////////////////////////

        void BuildBuffers()
        {
            //buffer for the 'spatial' dimensions of the array
            fSpatialDimensionBufferCL =
            new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, NDIM*sizeof(unsigned int));

            //buffer for the FFT twiddle factors
            fTwiddleBufferCL =
            new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, fMaxBufferSize*sizeof(CL_TYPE2));

            //buffer for the conjugate FFT twiddle factors
            fConjugateTwiddleBufferCL =
            new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, fMaxBufferSize*sizeof(CL_TYPE2));

            //buffer for the bluestein scale factors
            fScaleBufferCL =
            new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, fMaxBufferSize*sizeof(CL_TYPE2));

            //buffer for the bluestein circulant vector
            fCirculantBufferCL =
            new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, fMaxBufferSize*sizeof(CL_TYPE2));

            //buffer for the data to be transformed
            fDataBufferCL =
            new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_WRITE, fTotalDataSize*sizeof(CL_TYPE2));

            //buffer for the permutation_array
            fPermuationArrayCL =
            new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, fMaxBufferSize*sizeof(unsigned int));

        }

////////////////////////////////////////////////////////////////////////////////

        void AssignBuffers()
        {
            cl::CommandQueue& Q = KOpenCLInterface::GetInstance()->GetQueue();

            //assign buffers and set kernel arguments
            unsigned int n_multdim_ffts = fDimensionSize[0];
            //small arguments that do not need to be enqueued/written
            fFFTKernel->setArg(0, n_multdim_ffts); //number of multidimensional fft's to perform


            //assign buffers and set kernel arguments
            fFFTKernel->setArg(1, 0); //(index of the selected dimension) updated at each stage

            //assign buffers and set kernel arguments
            fFFTKernel->setArg(2, 0); //updated at each execution

            //array dimensionality written once
            fFFTKernel->setArg(3, *fSpatialDimensionBufferCL);
            Q.enqueueWriteBuffer(*fSpatialDimensionBufferCL, CL_TRUE, 0, NDIM*sizeof(unsigned int), fSpatialDim);


            //following are updated at each stage when necessary
            //however in the special case where all spatial dimensions are the same
            //we can write them to the GPU now and need not re-send them during execution
            fFFTKernel->setArg(4, *fTwiddleBufferCL);
            fFFTKernel->setArg(5, *fConjugateTwiddleBufferCL);
            fFFTKernel->setArg(6, *fScaleBufferCL);
            fFFTKernel->setArg(7, *fCirculantBufferCL);
            fFFTKernel->setArg(8, *fPermuationArrayCL);


            if(fAllSpatialDimensionsAreEqual)
            {
                //write the constant buffers
                Q.enqueueWriteBuffer(*fTwiddleBufferCL, CL_TRUE, 0, fMaxBufferSize*sizeof(CL_TYPE2), &(fTwiddle[0][0]) );
                Q.enqueueWriteBuffer(*fConjugateTwiddleBufferCL, CL_TRUE, 0, fMaxBufferSize*sizeof(CL_TYPE2), &(fConjugateTwiddle[0][0]) );
                Q.enqueueWriteBuffer(*fScaleBufferCL, CL_TRUE, 0, fMaxBufferSize*sizeof(CL_TYPE2), &(fScale[0][0]) );
                Q.enqueueWriteBuffer(*fCirculantBufferCL, CL_TRUE, 0, fMaxBufferSize*sizeof(CL_TYPE2), &(fCirculant[0][0]) );

                Q.enqueueWriteBuffer(*fPermuationArrayCL, CL_TRUE, 0, fMaxBufferSize*sizeof(unsigned int), &(fPermuationArray[0][0]) );
            }

            //the data is updated once per execution
            fFFTKernel->setArg(9, *fDataBufferCL);
        }

////////////////////////////////////////////////////////////////////////////////

        void FillDataBuffer()
        {
            if(fFillFromHostData)
            {
                //initialize data on the GPU from host memory
                cl::CommandQueue& Q = KOpenCLInterface::GetInstance()->GetQueue();
                CL_TYPE2* ptr = (CL_TYPE2*) ( &( (this->fInput->GetData())[0] ) );
                Q.enqueueWriteBuffer(*fDataBufferCL, CL_TRUE, 0, fTotalDataSize*sizeof(CL_TYPE2), ptr );
            }

            //otherwise assume data is already on GPU from previous result
        }

////////////////////////////////////////////////////////////////////////////////

        void ReadOutDataBuffer()
        {
            if(fReadOutDataToHost)
            {
                //read out data from the GPU to the host memory
                cl::CommandQueue& Q = KOpenCLInterface::GetInstance()->GetQueue();
                CL_TYPE2* ptr = (CL_TYPE2*) ( &( (this->fInput->GetData())[0] ) );
                Q.enqueueReadBuffer(*fDataBufferCL, CL_TRUE, 0, fTotalDataSize*sizeof(CL_TYPE2), ptr);
            }
            //otherwise assume we want to leave the data on the GPU for further processing
        }

////////////////////////////////////////////////////////////////////////////////

        virtual bool DoInputOutputDimensionsMatch()
        {
            unsigned int in[NDIM+1];
            unsigned int out[NDIM+1];

            this->fInput->GetArrayDimensions(in);
            this->fOutput->GetArrayDimensions(out);

            for(unsigned int i=0; i<NDIM+1; i++)
            {
                if(in[i] != out[i])
                {
                    return false;
                }
            }
            return true;
        }

////////////////////////////////////////////////////////////////////////////////

        bool fIsValid;
        bool fIsForward;
        bool fInitialized;
        bool fAllSpatialDimensionsAreEqual;
        bool fFillFromHostData;
        bool fReadOutDataToHost;
        unsigned int fDimensionSize[NDIM+1];
        unsigned int fSpatialDim[NDIM];

        unsigned int fMaxBufferSize;
        unsigned int fTotalDataSize;

        ////////////////////////////////////////////////////////////////////////
        //Workspace for needed coefficients
        std::vector< std::complex<double> > fTwiddle[NDIM];
        std::vector< std::complex<double> > fConjugateTwiddle[NDIM];
        std::vector< std::complex<double> > fScale[NDIM];
        std::vector< std::complex<double> > fCirculant[NDIM];
        std::vector< unsigned int > fPermuationArray[NDIM];

        ////////////////////////////////////////////////////////////////////////

        std::string fOpenCLFlags;

        mutable cl::Kernel* fFFTKernel;

        //buffer for the spatial dimensions of each block to be transformed
        cl::Buffer* fSpatialDimensionBufferCL;

        //buffer for the FFT twiddle factors
        cl::Buffer* fTwiddleBufferCL;

        //buffer for the conjugate FFT twiddle factors
        cl::Buffer* fConjugateTwiddleBufferCL;

        //buffer for the bluestein scale factors
        cl::Buffer* fScaleBufferCL;

        //buffer for the bluestein circulant vector
        cl::Buffer* fCirculantBufferCL;

        //buffer for the data to be transformed
        cl::Buffer* fDataBufferCL;

        //buffer for the permutation array
        cl::Buffer* fPermuationArrayCL;

        unsigned int fNLocal;
        unsigned int fNGlobal;

        ////////////////////////////////////////////////////////////////////////
};


}

#endif /* KFMBatchedMultidimensionalFastFourierTransform_OpenCL_H__ */
