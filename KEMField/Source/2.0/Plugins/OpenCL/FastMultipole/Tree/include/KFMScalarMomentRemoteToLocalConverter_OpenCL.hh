#ifndef KFMScalarMomentRemoteToLocalConverter_OpenCL_H__
#define KFMScalarMomentRemoteToLocalConverter_OpenCL_H__

#include "KOpenCLInterface.hh"
#include "KOpenCLKernelBuilder.hh"

#include "KFMScalarMomentRemoteToLocalConverter.hh"
#include "KFMBatchedMultidimensionalFastFourierTransform_OpenCL.hh"



namespace KEMField{

/**
*
*@file KFMScalarMomentRemoteToLocalConverter_OpenCL.hh
*@class KFMScalarMomentRemoteToLocalConverter_OpenCL
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Oct 12 13:24:38 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/


template< typename ObjectTypeList, typename SourceScalarMomentType, typename TargetScalarMomentType, typename KernelType, size_t SpatialNDIM >
class KFMScalarMomentRemoteToLocalConverter_OpenCL:
public KFMScalarMomentRemoteToLocalConverter<ObjectTypeList, SourceScalarMomentType, TargetScalarMomentType, KernelType, SpatialNDIM >
{
    public:

        typedef KFMScalarMomentRemoteToLocalConverter<ObjectTypeList, SourceScalarMomentType, TargetScalarMomentType, KernelType, SpatialNDIM >
        KFMScalarMomentRemoteToLocalConverterBaseType;

        KFMScalarMomentRemoteToLocalConverter_OpenCL():
            KFMScalarMomentRemoteToLocalConverterBaseType(),
            fConvolveKernel(NULL),
            fScaleKernel(NULL),
            fAddKernel(NULL),
            fM2LCoeffBufferCL(NULL),
            fNM2LBuffers(0),
            fWorkspaceBufferCL(NULL),
            fSourceScaleFactorArray(NULL),
            fTargetScaleFactorArray(NULL),
            fSourceScaleFactorBufferCL(NULL),
            fTargetScaleFactorBufferCL(NULL)
        {
            this->fDFTCalcOpenCL = new KFMBatchedMultidimensionalFastFourierTransform_OpenCL<SpatialNDIM>();
            this->fInitialized = false;
        }

        virtual ~KFMScalarMomentRemoteToLocalConverter_OpenCL()
        {
            delete fDFTCalcOpenCL;
            delete fConvolveKernel;
            delete fScaleKernel;
            delete fAddKernel;

            for(unsigned int i = 0; i<fNM2LBuffers; i++)
            {
                delete fM2LCoeffBufferCL[i];
            }
            delete[] fM2LCoeffBufferCL;

            delete fWorkspaceBufferCL;

            delete[] fSourceScaleFactorArray;
            delete[] fTargetScaleFactorArray;
            delete fSourceScaleFactorBufferCL;
            delete fTargetScaleFactorBufferCL;
        }


        virtual void Initialize()
        {

            if( !(this->fInitialized) )
            {
                CheckDeviceProperites();

                KFMScalarMomentRemoteToLocalConverterBaseType::Initialize(); //m2l coeff are calculated here

                if(this->fNTerms != 0 && this->fDim != 0)
                {
                    //intialize DFT calculator for array dimensions
                    fDFTCalcOpenCL->SetInput(this->fAllMultipoles);
                    fDFTCalcOpenCL->SetOutput(this->fAllLocalCoeff);

                    //all enqueue read/write buffers occur externel to the DFT kernel execution
                    fDFTCalcOpenCL->Initialize();

                    ConstructConvolutionKernel();
                    ConstructScaleKernel();
                    ConstructAddKernel();
                    BuildBuffers();
                    AssignBuffers();
                }
            }
        }


        virtual void ApplyAction(KFMNode<ObjectTypeList>* node)
        {
            if(node != NULL)
            {
                if( node->HasChildren() && this->ChildrenHaveNonZeroMoments(node) )
                {
                    //collect the multipoles
                    this->fCollector->ApplyAction(node);

                    //enqueue write the multipoles to the GPU
                    size_t moment_size = (this->fNTerms)*(this->fTotalSpatialSize);
                    EnqueueWriteMultipoleMoments();

                    if(this->fIsScaleInvariant)
                    {
                        unsigned int level = node->GetLevel() + 1;
                        fScaleKernel->setArg(3, level);
                    }

                    //if we have a scale invariant kernel, once we have computed the kernel reponse array once
                    //we only have to re-scale the moments, we don't have to recompute the array at each tree level
                    //any recomputation of the kernel reponse array for non-invariant kernels must be managed by an external class

                     //compute size of the array
                    unsigned int n_global = (this->fNTerms)*(this->fTotalSpatialSize);

                    //rescale the multipoles
                    if(this->fIsScaleInvariant)
                    {
                        //pad out n-global to be a multiple of the n-local
                        unsigned int nDummy = fNScaleLocal - (n_global%fNScaleLocal);
                        if(nDummy == fNScaleLocal){nDummy = 0;};
                        n_global += nDummy;
                        cl::NDRange global(n_global);
                        cl::NDRange local(fNScaleLocal);

                        //set the scale factor argument
                        fScaleKernel->setArg(4, *fSourceScaleFactorBufferCL);
                        //now enqueue the kernel
                        KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fScaleKernel, cl::NullRange, global, local);
                    }

                    //convolve the multipoles with the response functions to get local coeff
                    Convolve();

                    //if we have a scale invariant kernel
                    //factor the local coefficients depending on the tree level
                    if(this->fIsScaleInvariant)
                    {
                        //pad out n-global to be a multiple of the n-local
                        unsigned int nDummy = fNScaleLocal - (n_global%fNScaleLocal);
                        if(nDummy == fNScaleLocal){nDummy = 0;};
                        n_global += nDummy;
                        cl::NDRange global(n_global);
                        cl::NDRange local(fNScaleLocal);

                        //set scale factor argument
                        fScaleKernel->setArg(4, *fTargetScaleFactorBufferCL);
                        //now enqueue the kernel
                        KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fScaleKernel, cl::NullRange, global, local);
                        #ifdef ENFORCE_CL_FINISH
                        KOpenCLInterface::GetInstance()->GetQueue().finish();
                        #endif
                    }

                    //collect the original local coeff
                    this->CollectOriginalCoefficients(node);
                    EnqueueWriteOriginalLocalCoefficients();

                    //enqueue the pointwise addition kernel to add the contribution back to the orignal local coeff
                    //pad out n-global to be a multiple of the n-local
                    unsigned int nDummy = fNAddLocal - (n_global%fNAddLocal);
                    if(nDummy == fNAddLocal){nDummy = 0;};
                    n_global += nDummy;
                    cl::NDRange global(n_global);
                    cl::NDRange local(fNAddLocal);

                    //now enqueue the kernel
                    KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fAddKernel, cl::NullRange, global, local);
                    #ifdef ENFORCE_CL_FINISH
                    KOpenCLInterface::GetInstance()->GetQueue().finish();
                    #endif

                    //equeue read the local coeff from the GPU
                    KOpenCLInterface::GetInstance()->GetQueue().enqueueReadBuffer(*fFFTDataBufferCL, CL_TRUE, 0, moment_size*sizeof(CL_TYPE2), this->fPtrLocalCoeff);
                    #ifdef ENFORCE_CL_FINISH
                    KOpenCLInterface::GetInstance()->GetQueue().finish();
                    #endif

                    //distribute the local coefficients
                    this->DistributeCoefficients(node);
                }
            }
        }


    protected:

////////////////////////////////////////////////////////////////////////////////

        virtual void Convolve()
        {
            //first perform the forward dft on all the multipole coefficients
            fDFTCalcOpenCL->SetForward();
            fDFTCalcOpenCL->SetWriteOutHostDataFalse();
            fDFTCalcOpenCL->SetReadOutDataToHostFalse();
            fDFTCalcOpenCL->SetInput(this->fAllMultipoles);
            fDFTCalcOpenCL->SetOutput(this->fAllMultipoles);
            fDFTCalcOpenCL->ExecuteOperation();

            PointwiseMultiplyAndAddOpenCL();

            //now perform an inverse DFT on the x-formed local
            //coefficients to get the actual local coeff
            fDFTCalcOpenCL->SetWriteOutHostDataFalse();
            fDFTCalcOpenCL->SetReadOutDataToHostFalse();
            fDFTCalcOpenCL->SetInput(this->fAllLocalCoeff);
            fDFTCalcOpenCL->SetOutput(this->fAllLocalCoeff);
            fDFTCalcOpenCL->SetBackward();
            fDFTCalcOpenCL->ExecuteOperation();
        }

////////////////////////////////////////////////////////////////////////////////

        virtual void BuildBuffers()
        {
            size_t max_buffer_size = KOpenCLInterface::GetInstance()->GetDevice().getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
            size_t total_mem_size =  KOpenCLInterface::GetInstance()->GetDevice().getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();

            //size of the response functions
            size_t m2l_size = (this->fNTerms)*(this->fNTerms)*(this->fTotalSpatialSize);

            if( m2l_size*sizeof(CL_TYPE2) > max_buffer_size )
            {
                //we cannot fit the response functions within a single buffer

                if(m2l_size*sizeof(CL_TYPE2) > total_mem_size)
                {
                    //we cannot fit response_functions entirely on the gpu
                    //even if we use multiple buffers
                    size_t size_to_alloc_mb = ( m2l_size*sizeof(CL_TYPE2) )/(1024*1024);
                    size_t max_size_mb = max_buffer_size/(1024*1024);
                    size_t total_size_mb = total_mem_size/(1024*1024);

                    kfmout<<"KFMScalarMomentRemoteToLocalConverter::BuildBuffers: Error. Cannot allocate buffer of size: "<<size_to_alloc_mb<<" MB on a device with max allowable buffer size of: "<<max_size_mb<<" MB and total device memory of: "<<total_size_mb<<" MB."<<kfmendl;
                    kfmexit(1);
                }
                else
                {
                    //we might be able to fit the response functions on the gpu in multiple
                    //buffers, lets compute how many buffers we will need

                    size_t n_buffers_needed = std::ceil( ( (double)( m2l_size*sizeof(CL_TYPE2) ))/( (double)max_buffer_size)) + 1;

                    fNM2LBuffers = n_buffers_needed;

                    size_t n_terms_per_buffer = (this->fNTerms)/n_buffers_needed;
                    fNTermsPerBuffer = n_terms_per_buffer;

                    //allocate the buffer points
                    fM2LCoeffBufferCL = new cl::Buffer*[fNM2LBuffers];

                    //now allocate the buffers
                    size_t n_terms_left = this->fNTerms;
                    for(unsigned int i=0; i<fNM2LBuffers; i++)
                    {
                        if( i+1 != fNM2LBuffers)
                        {
                            size_t elements_in_buffer = fNTermsPerBuffer*(this->fNTerms)*(this->fTotalSpatialSize);
                            //create the m2l buffer
                            CL_ERROR_TRY
                            {
                                fM2LCoeffBufferCL[i]
                                = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, elements_in_buffer*sizeof(CL_TYPE2));
                            }
                            CL_ERROR_CATCH
                            n_terms_left -= fNTermsPerBuffer;
                        }
                        else
                        {
                            size_t elements_in_buffer = n_terms_left*(this->fNTerms)*(this->fTotalSpatialSize);
                            //create the m2l buffer
                            CL_ERROR_TRY
                            {
                                fM2LCoeffBufferCL[i]
                                = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, elements_in_buffer*sizeof(CL_TYPE2));
                            }
                            CL_ERROR_CATCH
                        }
                    }
                }
            }
            else
            {
                fNM2LBuffers = 1;
                fNTermsPerBuffer = this->fNTerms;
                fM2LCoeffBufferCL = new cl::Buffer*[fNM2LBuffers];

                //create the m2l buffer
                CL_ERROR_TRY
                {
                    fM2LCoeffBufferCL[0]
                    = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, m2l_size*sizeof(CL_TYPE2));
                }
                CL_ERROR_CATCH
            }

            //size of the workspace on the gpu
            size_t workspace_size = (this->fNTerms)*(this->fTotalSpatialSize);

            //create the workspace buffer
            CL_ERROR_TRY
            {
                this->fWorkspaceBufferCL
                = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_WRITE, workspace_size*sizeof(CL_TYPE2));
            }
            CL_ERROR_CATCH

            //get the pointer to the DFT calculators GPU data buffer
            //we will use this to directly fill the buffer with the multipoles, and local coefficients
            //for FFTs while it is still on the GPU
            this->fFFTDataBufferCL = this->fDFTCalcOpenCL->GetDataBuffer();

            if(this->fIsScaleInvariant)
            {
                //create the scale factor arrays
                size_t sf_size = (this->fMaxTreeDepth + 1)*(this->fNTerms);
                fSourceScaleFactorArray = new CL_TYPE[sf_size];
                fTargetScaleFactorArray = new CL_TYPE[sf_size];

                //fill them with the scale factors
                double level_side_length = this->fLength;
                double div_power;

                for(size_t level = 0; level <= this->fMaxTreeDepth; level++)
                {
                    div_power = this->fDiv;
                    if(level == 0){div_power = 1.0;};
                    if(level == 1){div_power = this->fTopLevelDivisions;};

                    level_side_length /= div_power;

                    //recompute the scale factors
                    std::complex<double> factor(level_side_length, 0.0);
                    for(size_t si=0; si<this->fNTerms; si++)
                    {
                        std::complex<double> s;
                        //compute the needed re-scaling for this tree level
                        s = this->fScaleInvariantKernel->GetSourceScaleFactor(si, factor );
                        fSourceScaleFactorArray[level*(this->fNTerms) + si] = std::real(s);

                        s = this->fScaleInvariantKernel->GetTargetScaleFactor(si, factor );
                        fTargetScaleFactorArray[level*(this->fNTerms) + si] = std::real(s);
                    }
                }

                //create the scale factor buffers
                CL_ERROR_TRY
                {
                    fSourceScaleFactorBufferCL
                    = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, sf_size*sizeof(CL_TYPE));
                }
                CL_ERROR_CATCH

                CL_ERROR_TRY
                {
                fTargetScaleFactorBufferCL
                = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, sf_size*sizeof(CL_TYPE));
                }
                CL_ERROR_CATCH

                //write the scale factors to the gpu
                KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fSourceScaleFactorBufferCL, CL_TRUE, 0, sf_size*sizeof(CL_TYPE), fSourceScaleFactorArray);
                #ifdef ENFORCE_CL_FINISH
                KOpenCLInterface::GetInstance()->GetQueue().finish();
                #endif

                //write the scale factors to the gpu
                KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fTargetScaleFactorBufferCL, CL_TRUE, 0, sf_size*sizeof(CL_TYPE), fTargetScaleFactorArray);
                #ifdef ENFORCE_CL_FINISH
                KOpenCLInterface::GetInstance()->GetQueue().finish();
                #endif
            }
        }

////////////////////////////////////////////////////////////////////////////////

        void PointwiseMultiplyAndAddOpenCL()
        {

            //now enqueue the kernel
            unsigned int array_size = this->fNTerms*(this->fTotalSpatialSize);

            //KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fConvolveKernel, cl::NullRange, global, local);

            //now enqueue the kernel for however many times we need to finish multiplication by response functions
            unsigned int n_terms_left = this->fNTerms;
            unsigned int n_terms_completed = 0;
            for(unsigned int i=0; i<fNM2LBuffers; i++)
            {
                if( i+1 != fNM2LBuffers)
                {
                    //compute size of the array
                    unsigned int n_global = fNTermsPerBuffer*(this->fTotalSpatialSize);

                    //set the array size
                    fConvolveKernel->setArg(0, n_global);

                    //pad out n-global to be a multiple of the n-local
                    unsigned int nDummy = fNConvolveLocal - (n_global%fNConvolveLocal);
                    if(nDummy == fNConvolveLocal){nDummy = 0;};
                    n_global += nDummy;

                    cl::NDRange global(n_global);
                    cl::NDRange local(fNConvolveLocal);

                    //set the target index offset
                    fConvolveKernel->setArg(3, n_terms_completed);

                    //set appropriate response function buffer
                    fConvolveKernel->setArg(5, *(fM2LCoeffBufferCL[i]) );

                    KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fConvolveKernel, cl::NullRange, global, local);
                    #ifdef ENFORCE_CL_FINISH
                    KOpenCLInterface::GetInstance()->GetQueue().finish();
                    #endif

                    n_terms_left -= fNTermsPerBuffer;
                    n_terms_completed += fNTermsPerBuffer;
                }
                else
                {
                    //compute size of the array
                    unsigned int n_global = n_terms_left*(this->fTotalSpatialSize);

                    //set the array size
                    fConvolveKernel->setArg(0, n_global);

                    //pad out n-global to be a multiple of the n-local
                    unsigned int nDummy = fNConvolveLocal - (n_global%fNConvolveLocal);
                    if(nDummy == fNConvolveLocal){nDummy = 0;};
                    n_global += nDummy;

                    cl::NDRange global(n_global);
                    cl::NDRange local(fNConvolveLocal);

                    //set the target index offset
                    fConvolveKernel->setArg(3, n_terms_completed);

                    //set appropriate response function buffer
                    fConvolveKernel->setArg(5, *(fM2LCoeffBufferCL[i]) );

                    KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fConvolveKernel, cl::NullRange, global, local);
                    #ifdef ENFORCE_CL_FINISH
                    KOpenCLInterface::GetInstance()->GetQueue().finish();
                    #endif
                }
            }

            //now copy the workspace data (pre-x-formed local) into the FFT buffer to perform inverse DFT
            KOpenCLInterface::GetInstance()->GetQueue().enqueueCopyBuffer(*fWorkspaceBufferCL, *fFFTDataBufferCL, size_t(0), size_t(0), array_size*sizeof(CL_TYPE2) );
            #ifdef ENFORCE_CL_FINISH
            KOpenCLInterface::GetInstance()->GetQueue().finish();
            #endif

        }


        void AssignBuffers()
        {
            unsigned int array_size = (this->fNTerms)*(this->fTotalSpatialSize);
            //(arg 0) the size of the array to process is assigned at kernel execution time

            //assign the array's spatial stride
            unsigned int spatial_stride = this->fTotalSpatialSize;
            fConvolveKernel->setArg(1, spatial_stride);

            //assign the total number of terms in an expansion
            unsigned int n_global_terms = this->fNTerms;
            fConvolveKernel->setArg(2, n_global_terms);

            //(arg 3) the target index offset is assigned at kernel execution time

            //assign the remote moments buffer
            fConvolveKernel->setArg(4, *fFFTDataBufferCL);

            //reponse function buffer (arg 5) is assigned at kernel execution time

            //assign the local moment buffer
            fConvolveKernel->setArg(6, *fWorkspaceBufferCL);

            fScaleKernel->setArg(0, array_size );
            fScaleKernel->setArg(1, spatial_stride);
            fScaleKernel->setArg(2, (unsigned int) this->fNTerms);
            fScaleKernel->setArg(3, (unsigned int) this->fMaxTreeDepth);
            fScaleKernel->setArg(4, *fSourceScaleFactorBufferCL);
            fScaleKernel->setArg(5, *fFFTDataBufferCL);

            fAddKernel->setArg(0, array_size);
            fAddKernel->setArg(1, *fFFTDataBufferCL);
            fAddKernel->setArg(2, *fWorkspaceBufferCL);
            fAddKernel->setArg(3, *fFFTDataBufferCL);

            //write the M2L coefficients to the GPU

            //now fill the buffers
            size_t n_terms_left = this->fNTerms;
            size_t n_terms_completed = 0;
            for(unsigned int i=0; i<fNM2LBuffers; i++)
            {
                if( i+1 != fNM2LBuffers)
                {
                    size_t elements_in_buffer = fNTermsPerBuffer*(this->fNTerms)*(this->fTotalSpatialSize);
                    std::complex<double>* ptr = &(this->fPtrM2LCoeff[(n_terms_completed*(this->fNTerms))*(this->fTotalSpatialSize)] );
                    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*(fM2LCoeffBufferCL[i]), CL_TRUE, 0, elements_in_buffer*sizeof(CL_TYPE2), ptr);
                    #ifdef ENFORCE_CL_FINISH
                    KOpenCLInterface::GetInstance()->GetQueue().finish();
                    #endif
                    n_terms_left -= fNTermsPerBuffer;
                    n_terms_completed += fNTermsPerBuffer;
                }
                else
                {
                    size_t elements_in_buffer = n_terms_left*(this->fNTerms)*(this->fTotalSpatialSize);
                    std::complex<double>* ptr = &(this->fPtrM2LCoeff[(n_terms_completed*(this->fNTerms))*(this->fTotalSpatialSize)] );
                    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*(fM2LCoeffBufferCL[i]), CL_TRUE, 0, elements_in_buffer*sizeof(CL_TYPE2), ptr);
                    #ifdef ENFORCE_CL_FINISH
                    KOpenCLInterface::GetInstance()->GetQueue().finish();
                    #endif
                }
            }

            //no longer need a host side copy of the M2L coeff
            delete[] this->fPtrM2LCoeff;
            this->fPtrM2LCoeff = NULL;

            //TODO: add special handling if M2L are too big to fit on the GPU
        }

        virtual void ConstructConvolutionKernel()
        {
            //Get name of kernel source file
            std::stringstream clFile;
            clFile << KOpenCLInterface::GetInstance()->GetKernelPath() << "/kEMField_KFMScalarMomentRemoteToLocalConverter_kernel.cl";

            KOpenCLKernelBuilder k_builder;
            fConvolveKernel = k_builder.BuildKernel(clFile.str(), std::string("ScalarMomentRemoteToLocalConverter") );

            //get n-local
            fNConvolveLocal = fConvolveKernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(KOpenCLInterface::GetInstance()->GetDevice());

            unsigned int preferredWorkgroupMultiple = fConvolveKernel->getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(KOpenCLInterface::GetInstance()->GetDevice() );

            if(preferredWorkgroupMultiple < fNConvolveLocal)
            {
                fNConvolveLocal = preferredWorkgroupMultiple;
            }

        }

////////////////////////////////////////////////////////////////////////////////

        virtual void ConstructScaleKernel()
        {
            //Get name of kernel source file
            std::stringstream clFile;
            clFile << KOpenCLInterface::GetInstance()->GetKernelPath() << "/kEMField_KFMScalarMomentApplyScaleFactor_kernel.cl";

            KOpenCLKernelBuilder k_builder;
            fScaleKernel = k_builder.BuildKernel(clFile.str(), std::string("ScalarMomentApplyScaleFactor") );

            //get n-local
            fNScaleLocal = fScaleKernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(KOpenCLInterface::GetInstance()->GetDevice());

            unsigned int preferredWorkgroupMultiple = fScaleKernel->getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(KOpenCLInterface::GetInstance()->GetDevice() );

            if(preferredWorkgroupMultiple < fNScaleLocal)
            {
                fNScaleLocal = preferredWorkgroupMultiple;
            }
        }

////////////////////////////////////////////////////////////////////////////////

        virtual void ConstructAddKernel()
        {
            //Get name of kernel source file
            std::stringstream clFile;
            clFile << KOpenCLInterface::GetInstance()->GetKernelPath() << "/kEMField_KFMPointwiseComplexVectorAdd_kernel.cl";

            KOpenCLKernelBuilder k_builder;
            fAddKernel = k_builder.BuildKernel(clFile.str(), std::string("PointwiseComplexVectorAdd") );

            //get n-local
            fNAddLocal = fAddKernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(KOpenCLInterface::GetInstance()->GetDevice());

            unsigned int preferredWorkgroupMultiple = fAddKernel->getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(KOpenCLInterface::GetInstance()->GetDevice() );

            if(preferredWorkgroupMultiple < fNAddLocal)
            {
                fNAddLocal = preferredWorkgroupMultiple;
            }
        }

////////////////////////////////////////////////////////////////////////////////

        void EnqueueWriteMultipoleMoments()
        {
            size_t moment_size = (this->fNTerms)*(this->fTotalSpatialSize);
            KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fFFTDataBufferCL, CL_TRUE, 0, moment_size*sizeof(CL_TYPE2), this->fPtrMultipoles);
            #ifdef ENFORCE_CL_FINISH
            KOpenCLInterface::GetInstance()->GetQueue().finish();
            #endif
        }

////////////////////////////////////////////////////////////////////////////////

        void EnqueueWriteOriginalLocalCoefficients()
        {
            size_t moment_size = (this->fNTerms)*(this->fTotalSpatialSize);
            //enqueue write the original local coeff to the GPU
            KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fWorkspaceBufferCL, CL_TRUE, 0, moment_size*sizeof(CL_TYPE2), this->fPtrOrigLocalCoeff);
            #ifdef ENFORCE_CL_FINISH
            KOpenCLInterface::GetInstance()->GetQueue().finish();
            #endif
        }

////////////////////////////////////////////////////////////////////////////////
        virtual void CheckDeviceProperites()
        {
            size_t max_buffer_size = KOpenCLInterface::GetInstance()->GetDevice().getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
            size_t total_mem_size =  KOpenCLInterface::GetInstance()->GetDevice().getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();

            //size of the response functions
            size_t m2l_size = (this->fNTerms)*(this->fNTerms)*(this->fTotalSpatialSize);

            if(m2l_size*sizeof(CL_TYPE2) > total_mem_size)
            {
                //we cannot fit response_functions entirely on the gpu
                //even if we use multiple buffers
                size_t size_to_alloc_mb = ( m2l_size*sizeof(CL_TYPE2) )/(1024*1024);
                size_t max_size_mb = max_buffer_size/(1024*1024);
                size_t total_size_mb = total_mem_size/(1024*1024);

                kfmout<<"KFMScalarMomentRemoteToLocalConverter_OpenCL::CheckDeviceProperites: Error. Cannot allocate buffer of size: "<<size_to_alloc_mb<<" MB on a device with max allowable buffer size of: "<<max_size_mb<<" MB and total device memory of: "<<total_size_mb<<" MB."<<kfmendl;
                kfmexit(1);
            }
        }

////////////////////////////////////////////////////////////////////////////////

        KFMBatchedMultidimensionalFastFourierTransform_OpenCL<SpatialNDIM>* fDFTCalcOpenCL;

        mutable cl::Kernel* fConvolveKernel;
        unsigned int fNConvolveLocal;

        mutable cl::Kernel* fScaleKernel;
        unsigned int fNScaleLocal;

        mutable cl::Kernel* fAddKernel;
        unsigned int fNAddLocal;

        //need a buffer to store the M2L coefficients on the GPU
        cl::Buffer** fM2LCoeffBufferCL;
        unsigned int fNM2LBuffers;
        unsigned int fNTermsPerBuffer;

        //need a buffer to copy the multpole moments into,
        //and to read the local coefficients out from
        //this is the input buffer of the batched FFT calculator, it is not owned
        cl::Buffer* fFFTDataBufferCL; //must be p^2*total_spatial_size

        //need a buffer to store the local coefficients on the GPU
        //this is a temporary buffer that only needs to operated on by the GPU
        //we copy this buffer into the batched FFT calculators buffer before
        //the final FFT to obtain the local coefficients
        cl::Buffer* fWorkspaceBufferCL; //must be p^2*total_spatial_size

        //scale factor buffers for scale invariant kernels
        CL_TYPE* fSourceScaleFactorArray;
        CL_TYPE* fTargetScaleFactorArray;
        cl::Buffer* fSourceScaleFactorBufferCL;
        cl::Buffer* fTargetScaleFactorBufferCL;



////////////////////////////////////////////////////////////////////////////////



};

}//end of KEMField namespace


#endif /* __KFMScalarMomentRemoteToLocalConverter_OpenCL_H__ */
