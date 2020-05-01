#ifndef KFMScalarMomentRemoteToRemoteConverter_OpenCL_H__
#define KFMScalarMomentRemoteToRemoteConverter_OpenCL_H__


#include "KFMArrayScalarMultiplier.hh"
#include "KFMArrayWrapper.hh"
#include "KFMCube.hh"
#include "KFMKernelExpansion.hh"
#include "KFMKernelResponseArray.hh"
#include "KFMNodeActor.hh"
#include "KFMObjectRetriever.hh"
#include "KFMPointwiseArrayAdder.hh"
#include "KFMPointwiseArrayMultiplier.hh"
#include "KFMScalarMomentCollector.hh"
#include "KFMScalarMomentDistributor.hh"
#include "KFMScalarMomentInitializer.hh"
#include "KFMScaleInvariantKernelExpansion.hh"
#include "KOpenCLInterface.hh"
#include "KOpenCLKernelBuilder.hh"

#include <complex>
#include <vector>


namespace KEMField
{

/**
*
*@file KFMScalarMomentRemoteToRemoteConverter_OpenCL.hh
*@class KFMScalarMomentRemoteToRemoteConverter_OpenCL
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Oct 12 13:24:38 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<typename ObjectTypeList, typename ScalarMomentType, typename KernelType, size_t SpatialNDIM>
class KFMScalarMomentRemoteToRemoteConverter_OpenCL :
    public KFMScalarMomentRemoteToRemoteConverter<ObjectTypeList, ScalarMomentType, KernelType, SpatialNDIM>
{
  public:
    typedef KFMScalarMomentRemoteToRemoteConverter<ObjectTypeList, ScalarMomentType, KernelType, SpatialNDIM>
        KFMScalarMomentRemoteToRemoteConverterBaseType;

    KFMScalarMomentRemoteToRemoteConverter_OpenCL() : KFMScalarMomentRemoteToRemoteConverterBaseType()
    {
        fM2MCoeffBufferCL = NULL;
        fChildMomentBufferCL = NULL;
        fTransformedChildMomentBufferCL = NULL;
        fParentMomentBufferCL = NULL;
        fScaleFactorBufferCL = NULL;
        fScaleFactorArray = NULL;
        this->fInitialized = false;
    };

    virtual ~KFMScalarMomentRemoteToRemoteConverter_OpenCL()
    {
        delete fM2MCoeffBufferCL;
        delete fChildMomentBufferCL;
        delete fTransformedChildMomentBufferCL;
        delete fParentMomentBufferCL;
        delete fScaleFactorBufferCL;

        delete[] fScaleFactorArray;
    };


    ////////////////////////////////////////////////////////////////////////
    void Initialize()
    {
        if (!(this->fInitialized)) {
            KFMScalarMomentRemoteToRemoteConverterBaseType::Initialize();

            ConstructKernel();
            ConstructReductionKernel();
            BuildBuffers();
            AssignBuffers();

            //create array for scale factors
            fScaleFactorArray = new CL_TYPE[this->fNTerms];

            //create array for parents moment contribution from children
            fParentMomentContribution.resize(this->fNTerms);
        }
    }


    ////////////////////////////////////////////////////////////////////////
    virtual void ApplyAction(KFMNode<ObjectTypeList>* node)
    {
        if (node != NULL && node->HasChildren() && node->GetLevel() != 0) {
            //first check if this node has children with non-zero multipole moments
            if (this->ChildrenHaveNonZeroMoments(node)) {
                this->fCollector->ApplyAction(node);

                //if we have a scale invariant kernel, so upon having computed the kernel reponse array once
                //we only have to re-scale the moments, we don't have to recompute the array at each tree level
                //any recomputation of the kernel reponse array for non-invariant kernels must be managed by an external class
                double child_side_length =
                    KFMObjectRetriever<ObjectTypeList, KFMCube<SpatialNDIM>>::GetNodeObject(node->GetChild(0))
                        ->GetLength();

                //also need to compute the scale factors for the child moment contributions
                if (this->fIsScaleInvariant) {
                    //apply the needed re-scaling for this tree level
                    std::complex<double> scale = std::complex<double>(child_side_length, 0.0);
                    for (size_t si = 0; si < this->fNTerms; si++) {
                        fScaleFactorArray[si] =
                            std::real(this->fKernelResponse->GetKernel()->GetSourceScaleFactor(si, scale));
                    }
                }

                //enqueue write out the children's coefficients to the gpu
                size_t child_moment_size = (this->fNTerms) * (this->fTotalSpatialSize);
                KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fChildMomentBufferCL,
                                                                               CL_TRUE,
                                                                               0,
                                                                               child_moment_size * sizeof(CL_TYPE2),
                                                                               this->fPtrChildMoments);
#ifdef ENFORCE_CL_FINISH
                KOpenCLInterface::GetInstance()->GetQueue().finish();
#endif

                //enqueue write out the scale factors for the contributions
                KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fScaleFactorBufferCL,
                                                                               CL_TRUE,
                                                                               0,
                                                                               (this->fNTerms) * sizeof(CL_TYPE),
                                                                               fScaleFactorArray);
#ifdef ENFORCE_CL_FINISH
                KOpenCLInterface::GetInstance()->GetQueue().finish();
#endif


                //compute the down conversion of this node's local coefficients to it's children
                //by pointwise multiply and sum, run Kernel

                PointwiseMultiplyAndAddOpenCL();

                //read out the contribution to the parents moments
                KOpenCLInterface::GetInstance()->GetQueue().enqueueReadBuffer(*fParentMomentBufferCL,
                                                                              CL_TRUE,
                                                                              0,
                                                                              (this->fNTerms) * sizeof(CL_TYPE2),
                                                                              &(fParentMomentContribution[0]));
#ifdef ENFORCE_CL_FINISH
                KOpenCLInterface::GetInstance()->GetQueue().finish();
#endif

                //scale the parent moments contribution and add to any prexisiting moments
                //also need to compute the scale factors for the child moment contributions
                if (this->fIsScaleInvariant) {
                    //apply the needed re-scaling for this tree level
                    std::complex<double> scale = std::complex<double>(child_side_length, 0.0);
                    for (size_t tsi = 0; tsi < this->fNTerms; tsi++) {
                        fParentMomentContribution[tsi] *=
                            std::real(this->fKernelResponse->GetKernel()->GetTargetScaleFactor(tsi, scale));
                    }
                }

                //if the node has a prexisting expansion we add the collected child moments
                //otherwise we create a new expansion
                if (KFMObjectRetriever<ObjectTypeList, ScalarMomentType>::GetNodeObject(node) == NULL) {
                    this->fMomentInitializer->ApplyAction(node);
                }

                this->fTargetCoeff.SetMoments(&fParentMomentContribution);
                this->fMomentDistributor->SetExpansionToAdd(&(this->fTargetCoeff));
                this->fMomentDistributor->ApplyAction(node);
            }
        }
    }

  protected:
    virtual void BuildBuffers()
    {
        size_t max_buffer_size = KOpenCLInterface::GetInstance()->GetDevice().getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();

        //size of the response functions
        size_t m2m_size = (this->fNTerms) * (this->fNTerms) * (this->fTotalSpatialSize);

        if (m2m_size * sizeof(CL_TYPE2) > max_buffer_size) {
            kfmout << "KFMScalarMomentRemoteToRemoteConverter::BuildBuffers: Error. Cannot allocated buffer of size: "
                   << m2m_size * sizeof(CL_TYPE2) << " on device with max allowable buffer size of: " << max_buffer_size
                   << std::endl
                   << kfmendl;
            kfmexit(1);
        }


        //create the m2m buffer
        CL_ERROR_TRY
        {
            this->fM2MCoeffBufferCL = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
                                                     CL_MEM_READ_ONLY,
                                                     m2m_size * sizeof(CL_TYPE2));
        }
        CL_ERROR_CATCH


        //size of the parent node's moments is fNTerms, create parent moment buffer
        CL_ERROR_TRY
        {
            this->fParentMomentBufferCL = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
                                                         CL_MEM_READ_WRITE,
                                                         (this->fNTerms) * sizeof(CL_TYPE2));
        }
        CL_ERROR_CATCH

        //size of the scale factor buffer is fNTerms, create scale factor buffer
        CL_ERROR_TRY
        {
            this->fScaleFactorBufferCL = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
                                                        CL_MEM_READ_WRITE,
                                                        (this->fNTerms) * sizeof(CL_TYPE));
        }
        CL_ERROR_CATCH

        //size of the local coefficient buffer on the gpu
        size_t moment_buffer_size = (this->fNTerms) * (this->fTotalSpatialSize);

        //create the children's moment buffer
        CL_ERROR_TRY
        {
            this->fChildMomentBufferCL = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
                                                        CL_MEM_READ_WRITE,
                                                        moment_buffer_size * sizeof(CL_TYPE2));
        }
        CL_ERROR_CATCH

        //create the transfromed children's moment buffer
        CL_ERROR_TRY
        {
            this->fTransformedChildMomentBufferCL = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(),
                                                                   CL_MEM_READ_WRITE,
                                                                   moment_buffer_size * sizeof(CL_TYPE2));
        }
        CL_ERROR_CATCH
    }

    ////////////////////////////////////////////////////////////////////////////////

    void PointwiseMultiplyAndAddOpenCL()
    {
        //compute size of the array
        unsigned int n_global = (this->fNTerms) * (this->fTotalSpatialSize);

        //pad out n-global to be a multiple of the n-local
        unsigned int nDummy = fNLocal - (n_global % fNLocal);
        if (nDummy == fNLocal) {
            nDummy = 0;
        };
        n_global += nDummy;

        cl::NDRange global(n_global);
        cl::NDRange local(fNLocal);

        //now enqueue the kernel
        KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fKernel, cl::NullRange, global, local);
#ifdef ENFORCE_CL_FINISH
        KOpenCLInterface::GetInstance()->GetQueue().finish();
#endif

        //now run the reduction kernel
        n_global = this->fNTerms;

        //pad out n-global to be a multiple of the n-local
        nDummy = fNReductionLocal - (n_global % fNReductionLocal);
        if (nDummy == fNReductionLocal) {
            nDummy = 0;
        };
        n_global += nDummy;

        global = cl::NDRange(n_global);
        local = cl::NDRange(fNReductionLocal);

        KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fReductionKernel,
                                                                         cl::NullRange,
                                                                         global,
                                                                         local);
#ifdef ENFORCE_CL_FINISH
        KOpenCLInterface::GetInstance()->GetQueue().finish();
#endif
    }


    void AssignBuffers()
    {
        unsigned int array_size = (this->fNTerms) * (this->fTotalSpatialSize);
        fKernel->setArg(0, array_size);

        unsigned int spatial_stride = this->fTotalSpatialSize;
        fKernel->setArg(1, spatial_stride);

        unsigned int n_terms = this->fNTerms;
        fKernel->setArg(2, n_terms);

        fKernel->setArg(3, *fScaleFactorBufferCL);

        fKernel->setArg(4, *fChildMomentBufferCL);

        fKernel->setArg(5, *fM2MCoeffBufferCL);

        fKernel->setArg(6, *fTransformedChildMomentBufferCL);


        fReductionKernel->setArg(0, array_size);
        fReductionKernel->setArg(1, spatial_stride);
        fReductionKernel->setArg(2, n_terms);
        fReductionKernel->setArg(3, *fTransformedChildMomentBufferCL);
        fReductionKernel->setArg(4, *fParentMomentBufferCL);


        //write the M2M coefficients to the GPU
        size_t m2m_size = (this->fNTerms) * (this->fNTerms) * (this->fTotalSpatialSize);
        KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fM2MCoeffBufferCL,
                                                                       CL_TRUE,
                                                                       0,
                                                                       m2m_size * sizeof(CL_TYPE2),
                                                                       this->fPtrM2MCoeff);
#ifdef ENFORCE_CL_FINISH
        KOpenCLInterface::GetInstance()->GetQueue().finish();
#endif

        //no longer need a host side copy of the M2M coeff
        delete[] this->fPtrM2MCoeff;
        this->fPtrM2MCoeff = NULL;

        //TODO: add special handling if M2M coeff are too big to fit on the GPU
    }

    ////////////////////////////////////////////////////////////////////////////////

    virtual void ConstructKernel()
    {
        //Get name of kernel source file
        std::stringstream clFile;
        clFile << KOpenCLInterface::GetInstance()->GetKernelPath()
               << "/kEMField_KFMScalarMomentRemoteToRemoteConverter_kernel.cl";

        KOpenCLKernelBuilder k_builder;
        fKernel = k_builder.BuildKernel(clFile.str(), std::string("ScalarMomentRemoteToRemoteConverter"));

        //get n-local
        fNLocal = fKernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(KOpenCLInterface::GetInstance()->GetDevice());

        unsigned int preferredWorkgroupMultiple =
            fKernel->getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(
                KOpenCLInterface::GetInstance()->GetDevice());

        if (preferredWorkgroupMultiple < fNLocal) {
            fNLocal = preferredWorkgroupMultiple;
        }
    }

    ////////////////////////////////////////////////////////////////////////////////

    virtual void ConstructReductionKernel()
    {
        //Get name of kernel source file
        std::stringstream clFile;
        clFile << KOpenCLInterface::GetInstance()->GetKernelPath()
               << "/kEMField_KFMScalarMomentArrayReduction_kernel.cl";

        KOpenCLKernelBuilder k_builder;
        fReductionKernel = k_builder.BuildKernel(clFile.str(), std::string("ScalarMomentArrayReduction"));

        //get n-local
        fNReductionLocal =
            fReductionKernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(KOpenCLInterface::GetInstance()->GetDevice());

        unsigned int preferredWorkgroupMultiple =
            fReductionKernel->getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(
                KOpenCLInterface::GetInstance()->GetDevice());

        if (preferredWorkgroupMultiple < fNReductionLocal) {
            fNReductionLocal = preferredWorkgroupMultiple;
        }
    }


    ////////////////////////////////////////////////////////////////////////////////

    mutable cl::Kernel* fKernel;
    unsigned int fNLocal;

    mutable cl::Kernel* fReductionKernel;
    unsigned int fNReductionLocal;

    //need a buffer to store the L2L coefficients on the GPU
    cl::Buffer* fM2MCoeffBufferCL;  //must be n_terms*n_terms*total_spatial_size

    //buffer to compute child local coeff
    cl::Buffer* fChildMomentBufferCL;  //must be n_terms*total_spatial_size

    //temporary buffer to store transformed child moments
    cl::Buffer* fTransformedChildMomentBufferCL;

    //buffer to store the scaled parent coeff
    cl::Buffer* fParentMomentBufferCL;
    std::vector<std::complex<double>> fParentMomentContribution;

    //scale factor buffers for scale invariant kernels
    CL_TYPE* fScaleFactorArray;
    cl::Buffer* fScaleFactorBufferCL;


    ////////////////////////////////////////////////////////////////////////////////
};


}  // namespace KEMField


#endif /* __KFMScalarMomentRemoteToRemoteConverter_OpenCL_H__ */
