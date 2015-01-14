#ifndef KFMScalarMomentLocalToLocalConverter_OpenCL_H__
#define KFMScalarMomentLocalToLocalConverter_OpenCL_H__


#include <vector>
#include <complex>

#include "KFMCube.hh"

#include "KFMNodeActor.hh"
#include "KFMObjectRetriever.hh"

#include "KFMKernelResponseArray.hh"
#include "KFMKernelExpansion.hh"
#include "KFMScaleInvariantKernelExpansion.hh"

#include "KFMScalarMomentInitializer.hh"
#include "KFMScalarMomentCollector.hh"
#include "KFMScalarMomentDistributor.hh"

#include "KFMArrayWrapper.hh"
#include "KFMArrayScalarMultiplier.hh"
#include "KFMPointwiseArrayAdder.hh"
#include "KFMPointwiseArrayMultiplier.hh"

#include "KOpenCLInterface.hh"
#include "KOpenCLKernelBuilder.hh"


namespace KEMField{

/**
*
*@file KFMScalarMomentLocalToLocalConverter_OpenCL.hh
*@class KFMScalarMomentLocalToLocalConverter_OpenCL
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Oct 12 13:24:38 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/


template< typename ObjectTypeList, typename ScalarMomentType, typename KernelType, size_t SpatialNDIM >
class KFMScalarMomentLocalToLocalConverter_OpenCL:
public KFMScalarMomentLocalToLocalConverter<ObjectTypeList, ScalarMomentType, KernelType, SpatialNDIM >
{
    public:

        typedef KFMScalarMomentLocalToLocalConverter<ObjectTypeList, ScalarMomentType, KernelType, SpatialNDIM >
        KFMScalarMomentLocalToLocalConverterBaseType;

        KFMScalarMomentLocalToLocalConverter_OpenCL():KFMScalarMomentLocalToLocalConverterBaseType()
        {
            fL2LCoeffBufferCL = NULL;
            fLocalCoeffBufferCL = NULL;
            fParentCoeffBufferCL = NULL;
            fScaleFactorBufferCL = NULL;
            fScaleFactorArray = NULL;
        };


        virtual ~KFMScalarMomentLocalToLocalConverter_OpenCL()
        {
            delete fL2LCoeffBufferCL;
            delete fLocalCoeffBufferCL;
            delete fParentCoeffBufferCL;
            delete fScaleFactorBufferCL;
            delete[] fScaleFactorArray;
        };


        ////////////////////////////////////////////////////////////////////////
        void Initialize()
        {
            if( !(this->fInitialized) )
            {
                KFMScalarMomentLocalToLocalConverterBaseType::Initialize();

                ConstructKernel();
                BuildBuffers();
                AssignBuffers();

                //create array for scale factors
                fScaleFactorArray = new CL_TYPE[this->fNTerms];

            }
        }


        ////////////////////////////////////////////////////////////////////////
        virtual void ApplyAction(KFMNode<ObjectTypeList>* node)
        {
            if( node != NULL && node->HasChildren() )
            {
                ScalarMomentType* mom = KFMObjectRetriever<ObjectTypeList, ScalarMomentType>::GetNodeObject(node);
                if(mom != NULL)
                {
                    //collect the local coefficients of this node
                    mom->GetMoments(&(this->fParentsLocalCoeff));

                    //if we have a scale invariant kernel, so upon having computed the kernel reponse array once
                    //we only have to re-scale the moments, we don't have to recompute the array at each tree level
                    //any recomputation of the kernel reponse array for non-invariant kernels must be managed by an external class
                    double child_side_length =
                    KFMObjectRetriever<ObjectTypeList, KFMCube<SpatialNDIM> >::GetNodeObject(node->GetChild(0))->GetLength();
                    if(this->fIsScaleInvariant)
                    {
                        //apply the needed re-scaling for this tree level
                        std::complex<double> scale = std::complex<double>(child_side_length, 0.0);
                        for(size_t si=0; si<this->fNTerms; si++)
                        {
                            this->fPtrLocalCoeffSource[si]  = (this->fParentsLocalCoeff[si])*(this->fKernelResponse->GetKernel()->GetSourceScaleFactor(si, scale));
                        }
                    }


                    //also need to compute the scale factors for the child moment contributions
                    if(this->fIsScaleInvariant)
                    {
                        //apply the needed re-scaling for this tree level
                        std::complex<double> scale = std::complex<double>(child_side_length, 0.0);
                        for(size_t si=0; si<this->fNTerms; si++)
                        {
                            fScaleFactorArray[si]  = std::real( this->fKernelResponse->GetKernel()->GetTargetScaleFactor(si, scale) );
                        }
                    }

                    //enqueue write out the parent's coefficients to the gpu
                    size_t moment_size = (this->fNTerms);
                    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fParentCoeffBufferCL, CL_TRUE, 0, moment_size*sizeof(CL_TYPE2), this->fPtrLocalCoeffSource);

                    //enqueue write out the scale factors for the contributions
                    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fScaleFactorBufferCL, CL_TRUE, 0, moment_size*sizeof(CL_TYPE), this->fScaleFactorArray);

                    //collect the original local coefficients of the children of this node
                    this->fCollector->ApplyAction(node);

                    //enqueue write the existing local coeff to gpu
                    size_t local_coeff_size = (this->fNTerms)*(this->fTotalSpatialSize);
                    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fLocalCoeffBufferCL, CL_TRUE, 0, local_coeff_size*sizeof(CL_TYPE2), this->fPtrLocalCoeffOrig);

                    //compute the down conversion of this node's local coefficients to it's children
                    //by pointwise multiply and sum, run Kernel

                    PointwiseMultiplyAndAddOpenCL();

                    //read out the updated child local coeff
                    KOpenCLInterface::GetInstance()->GetQueue().enqueueReadBuffer(*fLocalCoeffBufferCL, CL_TRUE, 0, local_coeff_size*sizeof(CL_TYPE2), this->fPtrLocalCoeffOrig);

                    this->DistributeParentsCoefficients(node);

                }
            }
        }

    protected:

        virtual void BuildBuffers()
        {
            //first check that we have enough memory on the device for the response functions
            size_t max_buffer_size = KOpenCLInterface::GetInstance()->GetDevice().getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();

            //size of the response functions
            size_t l2l_size = (this->fNTerms)*(this->fNTerms)*(this->fTotalSpatialSize);

            if( l2l_size*sizeof(CL_TYPE2) > max_buffer_size )
            {
                kfmout<<"KFMScalarMomentLocalToLocalConverter::BuildBuffers: Error. Cannot allocated buffer of size: "<<l2l_size*sizeof(CL_TYPE2)<<" on device with max allowable buffer size of: "<<max_buffer_size<<std::endl<<kfmendl;
                kfmexit(1);
            }


            //create the l2l buffer
            this->fL2LCoeffBufferCL
            = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, l2l_size*sizeof(CL_TYPE2));

            //size of the parent node's coefficients is fNTerms, create parent moment buffer
            this->fParentCoeffBufferCL
            = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_WRITE, (this->fNTerms)*sizeof(CL_TYPE2));

            //size of the scale factor buffer is fNTerms, create scale factor buffer
            this->fScaleFactorBufferCL
            = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_WRITE, (this->fNTerms)*sizeof(CL_TYPE));

            //size of the local coefficient buffer on the gpu
            size_t local_coeff_buffer_size = (this->fNTerms)*(this->fTotalSpatialSize);

            //create the children's local coefficient buffer
            this->fLocalCoeffBufferCL
            = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_WRITE, local_coeff_buffer_size*sizeof(CL_TYPE2));
        }

////////////////////////////////////////////////////////////////////////////////

        void PointwiseMultiplyAndAddOpenCL()
        {
            //compute size of the array
            unsigned int n_global = (this->fNTerms)*(this->fTotalSpatialSize);

            //pad out n-global to be a multiple of the n-local
            unsigned int nDummy = fNLocal - (n_global%fNLocal);
            if(nDummy == fNLocal){nDummy = 0;};
            n_global += nDummy;

            cl::NDRange global(n_global);
            cl::NDRange local(fNLocal);

            //now enqueue the kernel
            KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fKernel, cl::NullRange, global, local);
        }


        void AssignBuffers()
        {
            unsigned int array_size = (this->fNTerms)*(this->fTotalSpatialSize);
            fKernel->setArg(0, array_size);

            unsigned int spatial_stride = this->fTotalSpatialSize;
            fKernel->setArg(1, spatial_stride);

            unsigned int n_terms = this->fNTerms;
            fKernel->setArg(2, n_terms);

            fKernel->setArg(3, *fScaleFactorBufferCL);
            fKernel->setArg(4, *fParentCoeffBufferCL);
            fKernel->setArg(5, *fL2LCoeffBufferCL);
            fKernel->setArg(6, *fLocalCoeffBufferCL);

            //write the L2L coefficients to the GPU
            size_t l2l_size = (this->fNTerms)*(this->fNTerms)*(this->fTotalSpatialSize);
            KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fL2LCoeffBufferCL, CL_TRUE, 0, l2l_size*sizeof(CL_TYPE2), this->fPtrL2LCoeff);

            //no longer need a host side copy of the L2L coeff
            delete[] this->fPtrL2LCoeff;
            this->fPtrL2LCoeff = NULL;

            //TODO: add special handling if L2L are too big to fit on the GPU
        }

        virtual void ConstructKernel()
        {
            //Get name of kernel source file
            std::stringstream clFile;
            clFile << KOpenCLInterface::GetInstance()->GetKernelPath() << "/kEMField_KFMScalarMomentLocalToLocalConverter_kernel.cl";

            KOpenCLKernelBuilder k_builder;
            fKernel = k_builder.BuildKernel(clFile.str(), std::string("ScalarMomentLocalToLocalConverter") );

            //get n-local
            fNLocal = fKernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(KOpenCLInterface::GetInstance()->GetDevice());
        }

////////////////////////////////////////////////////////////////////////////////

        mutable cl::Kernel* fKernel;
        unsigned int fNLocal;

        //need a buffer to store the L2L coefficients on the GPU
        cl::Buffer* fL2LCoeffBufferCL; //must be n_terms*n_terms*total_spatial_size

        //buffer to compute child local coeff
        cl::Buffer* fLocalCoeffBufferCL; //must be n_terms*total_spatial_size

        //buffer to store the scaled parent coeff
        cl::Buffer* fParentCoeffBufferCL;

        //scale factor buffers for scale invariant kernels
        CL_TYPE* fScaleFactorArray;
        cl::Buffer* fScaleFactorBufferCL;


////////////////////////////////////////////////////////////////////////////////
};


}



#endif /* __KFMScalarMomentLocalToLocalConverter_OpenCL_H__ */ 
