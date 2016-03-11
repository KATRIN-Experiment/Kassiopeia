#include "KFMElectrostaticRemoteToRemoteConverter_OpenCL.hh"

namespace KEMField
{



KFMElectrostaticRemoteToRemoteConverter_OpenCL::KFMElectrostaticRemoteToRemoteConverter_OpenCL()
{
    fTree = NULL;

    fCopyAndScaleKernel = NULL;
    fNCopyAndScaleLocal = 0;

    fTransformationKernel = NULL;
    fNTransformationLocal = 0;

    fReduceAndScaleKernel = NULL;
    fNReduceAndScaleLocal = 0;

    fM2MCoeffBufferCL = NULL;
    fChildMomentBufferCL = NULL;
    fTransformedChildMomentBufferCL = NULL;
    fSourceScaleFactorBufferCL = NULL;
    fTargetScaleFactorBufferCL = NULL;
    fNodeMomentBufferCL = NULL;
    fNodeIDBufferCL = NULL;
    fBlockSetIDListBufferCL = NULL;

    fNMultipoleNodes = 0;
    fMultipoleNodes = NULL;

    fCachedMultipoleNodeLists.clear();
    fCachedBlockSetIDLists.clear();

    fDegree = 0;
    fNTerms = 0;
    fStride = 0;
    fDivisions = 0;
    fTopLevelDivisions = 0;
    fLowerLevelDivisions = 0;
    fMaxTreeDepth = 0;
    fWorldLength = 0;

    fKernelResponse = new KFMKernelResponseArray_3DLaplaceM2M(); //false -> origin is the target
    fScaleInvariantKernel = NULL;

    fM2MCoeff = NULL;
};


KFMElectrostaticRemoteToRemoteConverter_OpenCL::~KFMElectrostaticRemoteToRemoteConverter_OpenCL()
{
    delete fCopyAndScaleKernel;
    delete fTransformationKernel;
    delete fReduceAndScaleKernel;

    delete fM2MCoeffBufferCL;
    delete fChildMomentBufferCL;
    delete fTransformedChildMomentBufferCL;
    delete fSourceScaleFactorBufferCL;
    delete fTargetScaleFactorBufferCL;
    delete fNodeIDBufferCL;
    delete fBlockSetIDListBufferCL;

    delete fKernelResponse;
    delete fM2MCoeff;
};

void
KFMElectrostaticRemoteToRemoteConverter_OpenCL::SetParameters(KFMElectrostaticParameters params)
{
    //set parameters
    fDegree = params.degree;
    fNTerms = (fDegree+1)*(fDegree+1);
    fStride = (fDegree+1)*(fDegree+2)/2;
    fDivisions = params.divisions;
    fMaxTreeDepth = params.maximum_tree_depth;
    fZeroMaskSize = params.zeromask;
}


void
KFMElectrostaticRemoteToRemoteConverter_OpenCL::SetTree(KFMElectrostaticTree* tree)
{
    fTree = tree;

    //determine world region size to compute scale factors
    KFMCube<KFMELECTROSTATICS_DIM>* world_cube =
    KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCube<KFMELECTROSTATICS_DIM> >::GetNodeObject(fTree->GetRootNode());
    fWorldLength = world_cube->GetLength();

    //now we want to retrieve the top level and lower level divisions
    //since we need both in order to compute the scale factor for the different tree levels correctly
    const unsigned int* dim_size;
    dim_size = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCubicSpaceTreeProperties<KFMELECTROSTATICS_DIM> >::GetNodeObject(fTree->GetRootNode())->GetTopLevelDimensions();
    fTopLevelDivisions = dim_size[0];

    dim_size = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCubicSpaceTreeProperties<KFMELECTROSTATICS_DIM> >::GetNodeObject(fTree->GetRootNode())->GetDimensions();
    fLowerLevelDivisions = dim_size[0];
};


void
KFMElectrostaticRemoteToRemoteConverter_OpenCL::SetMultipoleNodeSet(KFMSpecialNodeSet<KFMElectrostaticNodeObjects>* multipole_node_set)
{
    fMultipoleNodes = multipole_node_set;

    fNMultipoleNodes = fMultipoleNodes->GetSize();
    fCachedMultipoleNodeLists.resize(fNMultipoleNodes);
    fCachedBlockSetIDLists.resize(fNMultipoleNodes);

    //fill the node's list of child nodes and their positions (block set ids)
    for(unsigned int i=0; i<fNMultipoleNodes; i++)
    {
        fCachedMultipoleNodeLists[i].clear();
        fCachedBlockSetIDLists[i].clear();

        KFMNode<KFMElectrostaticNodeObjects>* node = fMultipoleNodes->GetNodeFromSpecializedID(i);

        if(node->HasChildren())
        {
            unsigned int n_children = node->GetNChildren();
            for(unsigned int j=0; j<n_children; j++)
            {
                int special_id = fMultipoleNodes->GetSpecializedIDFromOrdinaryID(node->GetChild(j)->GetID());

                if(special_id != -1)
                {
                    unsigned int id = static_cast<unsigned int>(special_id);
                    fCachedMultipoleNodeLists[i].push_back(id);
                    fCachedBlockSetIDLists[i].push_back(j);
                }
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////
void
KFMElectrostaticRemoteToRemoteConverter_OpenCL::Initialize()
{


    //set number of terms in series for response functions
    fKernelResponse->SetNumberOfTermsInSeries(fNTerms);

    fLowerLimits[0] = 0;
    fLowerLimits[1] = 0;
    fUpperLimits[0] = fNTerms;
    fUpperLimits[1] = fNTerms;
    fDimensionSize[0] = fNTerms;
    fDimensionSize[1] = fNTerms;

    for(unsigned int i=0; i<KFMELECTROSTATICS_DIM; i++)
    {
        fLowerLimits[i+2] = 0;
        fUpperLimits[i+2] = fDivisions;
        fDimensionSize[i+2] = fDivisions;
    }

    fTotalSpatialSize = KFMArrayMath::TotalArraySize<KFMELECTROSTATICS_DIM>( &(fDimensionSize[2]) );

    fKernelResponse->SetLowerSpatialLimits(&(fLowerLimits[2]));
    fKernelResponse->SetUpperSpatialLimits(&(fUpperLimits[2]));

    //set the source origin here...the position of the source
    //origin should be measured with respect to the center of the child node that is
    //indexed by (0,0,0), spacing between child nodes should be equal to 1.0
    //scaling for various tree levels is handled elsewhere

    double source_origin[KFMELECTROSTATICS_DIM];

    for(unsigned int i=0; i<KFMELECTROSTATICS_DIM; i++)
    {
        if(fDivisions%2 == 0)
        {
            source_origin[i] = 0.5;
        }
        else
        {
            source_origin[i] = 0.0;
        }
    }

    int shift[KFMELECTROSTATICS_DIM];
    for(unsigned int i=0; i<KFMELECTROSTATICS_DIM; i++)
    {
        shift[i] = -1*(std::ceil( 1.0*(((double)fDivisions)/2.0) ) - 1);
    }

    fKernelResponse->SetOrigin(source_origin);
    fKernelResponse->SetShift(shift);


    //allocate space and wrapper for M2M coeff
    fRawM2MCoeff.resize(fNTerms*fNTerms*fTotalSpatialSize);
    fM2MCoeff = new KFMArrayWrapper<std::complex<double>, KFMELECTROSTATICS_DIM + 2>( &(fRawM2MCoeff[0]), fDimensionSize);

    //here we need to initialize the M2M calculator
    //and fill the array of M2M coefficients
    fKernelResponse->SetZeroMaskSize(0);
    fKernelResponse->SetDistance(1.0);
    fKernelResponse->SetOutput(fM2MCoeff);
    fKernelResponse->Initialize();
    fKernelResponse->ExecuteOperation();

    //now compute the source and target scale factors
    fSourceScaleFactorArray.resize((fMaxTreeDepth+1)*fStride);
    fTargetScaleFactorArray.resize((fMaxTreeDepth+1)*fStride);

    //create the scale factor arrays
    //fill them with the scale factors
    double level_side_length = fWorldLength;
    double div_power;

    fScaleInvariantKernel = dynamic_cast< KFMScaleInvariantKernelExpansion<KFMELECTROSTATICS_DIM>* >( fKernelResponse->GetKernel() );

    for(size_t level = 0; level <= fMaxTreeDepth; level++)
    {
        div_power = (double)fLowerLevelDivisions;

        if(level == 0 ){div_power = 1.0;};
        if(level == 1 ){div_power = (double)fTopLevelDivisions;};

        level_side_length /= div_power;

        //recompute the scale factors
        std::complex<double> factor(level_side_length, 0.0);
        for(unsigned int n=0; n <= fDegree; n++)
        {
            for(unsigned int m=0; m <= n; m++)
            {
                unsigned int csi = KFMScalarMultipoleExpansion::ComplexBasisIndex(n,m);
                unsigned int rsi = KFMScalarMultipoleExpansion::RealBasisIndex(n,m);

                std::complex<double> s;
                //compute the needed re-scaling for this tree level
                s = fScaleInvariantKernel->GetSourceScaleFactor(csi, factor );
                fSourceScaleFactorArray[level*fStride + rsi] = std::real(s);

                s = fScaleInvariantKernel->GetTargetScaleFactor(csi, factor );
                fTargetScaleFactorArray[level*fStride + rsi] = std::real(s);
            }
        }
    }


    ConstructCopyAndScaleKernel();
    ConstructTransformationKernel();
    ConstructReduceAndScaleKernel();
    BuildBuffers();
    AssignBuffers();
}

void KFMElectrostaticRemoteToRemoteConverter_OpenCL::Prepare()
{
    //send the node multipole moments to the device
    unsigned int size = fNMultipoleNodes*fStride;

    std::vector< std::complex<CL_TYPE> > moments;
    moments.resize(size);

    //now distribute the primary node moments
    for(unsigned int i=0; i<fNMultipoleNodes; i++)
    {
        KFMNode<KFMElectrostaticNodeObjects>* node = fMultipoleNodes->GetNodeFromSpecializedID(i);
        KFMElectrostaticMultipoleSet* set = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMElectrostaticMultipoleSet>::GetNodeObject(node);

        if(set != NULL)
        {
            std::complex<CL_TYPE> temp;
            //we use raw ptr for speed
            double* rmoments = &( (*(set->GetRealMoments()))[0] );
            double* imoments = &( (*(set->GetImaginaryMoments()))[0] );
            for(unsigned int j=0; j < fStride; ++j)
            {
                temp.real(rmoments[j]);
                temp.imag(imoments[j]);
                moments[i*fStride + j] = temp;
            }
        }
    }

    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fNodeMomentBufferCL, CL_TRUE, 0, size*sizeof(CL_TYPE2), &(moments[0]) );

    #ifdef ENFORCE_CL_FINISH
    KOpenCLInterface::GetInstance()->GetQueue().finish();
    #endif

    moments.resize(0);
}

void KFMElectrostaticRemoteToRemoteConverter_OpenCL::Finalize()
{
    //read the node multipole coefficients back from the device
    unsigned int size = fNMultipoleNodes*fStride;

    std::vector< std::complex<CL_TYPE> > moments;
    moments.resize(size);

    KOpenCLInterface::GetInstance()->GetQueue().enqueueReadBuffer(*fNodeMomentBufferCL, CL_TRUE, 0, size*sizeof(CL_TYPE2), &(moments[0]) );

    //now distribute the primary node moments
    for(unsigned int i=0; i<fNMultipoleNodes; i++)
    {
        KFMNode<KFMElectrostaticNodeObjects>* node = fMultipoleNodes->GetNodeFromSpecializedID(i);
        KFMElectrostaticMultipoleSet* set = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMElectrostaticMultipoleSet>::GetNodeObject(node);

        if(set != NULL)
        {
            std::complex<double> temp;
            //we use raw ptr for speed
            double* rmoments = &( (*(set->GetRealMoments()))[0] );
            double* imoments = &( (*(set->GetImaginaryMoments()))[0] );
            for(unsigned int j=0; j < fStride; ++j)
            {
                temp = moments[i*fStride + j];
                rmoments[j] = temp.real();
                imoments[j] = temp.imag();
            }
        }
    }

    moments.resize(0);
}

////////////////////////////////////////////////////////////////////////
void
KFMElectrostaticRemoteToRemoteConverter_OpenCL::ApplyAction(KFMNode<KFMElectrostaticNodeObjects>* node)
{
    if( node != NULL && node->HasChildren() && node->GetLevel() != 0)
    {
        //check if this node is a member of the non-zero multipole node set
        int special_id = fMultipoleNodes->GetSpecializedIDFromOrdinaryID(node->GetID());

        if(special_id != -1)
        {
            //set the node id argument
            fReduceAndScaleKernel->setArg(3, static_cast<unsigned int>(special_id) );

            //get the node tree level and set the tree level arguments
            unsigned int tree_level = node->GetLevel() + 1;
            fCopyAndScaleKernel->setArg(2, tree_level);
            fReduceAndScaleKernel->setArg(2, tree_level);

            //get the number of children which have non-zero multipole moments
            //and et the number of moment sets argument
            unsigned int n_moment_sets = fCachedMultipoleNodeLists[special_id].size();

            if(n_moment_sets != 0)
            {

                fCopyAndScaleKernel->setArg(0, n_moment_sets);
                fTransformationKernel->setArg(0, n_moment_sets);
                fReduceAndScaleKernel->setArg(0, n_moment_sets);

                //write out this nodes child node id's
                KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fNodeIDBufferCL, CL_TRUE, 0, n_moment_sets*sizeof(unsigned int), &(fCachedMultipoleNodeLists[special_id][0]));

                #ifdef ENFORCE_CL_FINISH
                KOpenCLInterface::GetInstance()->GetQueue().finish();
                #endif

                //write out this nodes block set id's
                KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBlockSetIDListBufferCL, CL_TRUE, 0, n_moment_sets*sizeof(unsigned int), &(fCachedBlockSetIDLists[special_id][0]));

                #ifdef ENFORCE_CL_FINISH
                KOpenCLInterface::GetInstance()->GetQueue().finish();
                #endif

                unsigned int nDummy;
                unsigned int nLocal;
                unsigned int nGlobal;

                //run the copy and scale kernel
                nLocal = fNCopyAndScaleLocal;
                nGlobal = n_moment_sets;
                nDummy = nLocal - (nGlobal%nLocal);
                if(nDummy == nLocal){nDummy = 0;};
                KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fCopyAndScaleKernel, cl::NullRange, cl::NDRange(nGlobal + nDummy), cl::NDRange(nLocal) );

                #ifdef ENFORCE_CL_FINISH
                KOpenCLInterface::GetInstance()->GetQueue().finish();
                #endif

                //run the transformation kernel
                nLocal = fNTransformationLocal;
                nGlobal = n_moment_sets*fStride;
                nDummy = nLocal - (nGlobal%nLocal);
                if(nDummy == nLocal){nDummy = 0;};
                KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fTransformationKernel, cl::NullRange, cl::NDRange(nGlobal + nDummy), cl::NDRange(nLocal) );

                #ifdef ENFORCE_CL_FINISH
                KOpenCLInterface::GetInstance()->GetQueue().finish();
                #endif

                //run the reduce and scale kernel
                nLocal = fNReduceAndScaleLocal;
                nGlobal = fStride;
                nDummy = nLocal - (nGlobal%nLocal);
                if(nDummy == nLocal){nDummy = 0;};
                KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fReduceAndScaleKernel, cl::NullRange, cl::NDRange(nGlobal + nDummy), cl::NDRange(nLocal) );

                #ifdef ENFORCE_CL_FINISH
                KOpenCLInterface::GetInstance()->GetQueue().finish();
                #endif

            }

        }
    }
}

void
KFMElectrostaticRemoteToRemoteConverter_OpenCL::BuildBuffers()
{

//    fM2MCoeffBufferCL; !!!!
//    fTransformedChildMomentBufferCL; !!!!
//    fSourceScaleFactorBufferCL; !!!!
//    fTargetScaleFactorBufferCL; !!!!
//    fNodeMomentBufferCL; //not created here
//    fNodeIDBufferCL; !!!!
//    fBlockSetIDListBufferCL; !!!!

    size_t max_buffer_size = KOpenCLInterface::GetInstance()->GetDevice().getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();

    //size of the response functions
    size_t m2m_size = fNTerms*fNTerms*fTotalSpatialSize;

    if( m2m_size*sizeof(CL_TYPE2) > max_buffer_size )
    {
        //TODO: add special handling if M2M coeff are too big to fit on the GPU
        kfmout<<"KFMScalarMomentRemoteToRemoteConverter::BuildBuffers: Error. Cannot allocated buffer of size: "<<m2m_size*sizeof(CL_TYPE2)<<" on device with max allowable buffer size of: "<<max_buffer_size<<std::endl<<kfmendl;
        kfmexit(1);
    }

    //create the m2m buffer
    CL_ERROR_TRY
    {
        fM2MCoeffBufferCL
        = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, m2m_size*sizeof(CL_TYPE2));
    }
    CL_ERROR_CATCH

    //write the M2M coefficients to the GPU
    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fM2MCoeffBufferCL, CL_TRUE, 0, m2m_size*sizeof(CL_TYPE2),  &(fRawM2MCoeff[0]) );

    #ifdef ENFORCE_CL_FINISH
    KOpenCLInterface::GetInstance()->GetQueue().finish();
    #endif

    //no longer need a host side copy of the M2M coeff, swap them with empty vector
    std::vector< std::complex<double > > temp;
    temp.swap(fRawM2MCoeff);

    //create the untransformed but scaled children's moment buffer
    size_t moment_buffer_size = fStride*fTotalSpatialSize;
    CL_ERROR_TRY
    {
        fChildMomentBufferCL
        = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_WRITE, moment_buffer_size*sizeof(CL_TYPE2));
    }
    CL_ERROR_CATCH

    //create the transformed children's moment buffer
    CL_ERROR_TRY
    {
        fTransformedChildMomentBufferCL
        = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_WRITE, moment_buffer_size*sizeof(CL_TYPE2));
    }
    CL_ERROR_CATCH

    //create the scale factor buffers
    size_t sf_size =  ((fMaxTreeDepth+1)*fStride);
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
    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fSourceScaleFactorBufferCL, CL_TRUE, 0, sf_size*sizeof(CL_TYPE), &(fSourceScaleFactorArray[0]) );

    #ifdef ENFORCE_CL_FINISH
    KOpenCLInterface::GetInstance()->GetQueue().finish();
    #endif

    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fTargetScaleFactorBufferCL, CL_TRUE, 0, sf_size*sizeof(CL_TYPE), &(fTargetScaleFactorArray[0]) );

    #ifdef ENFORCE_CL_FINISH
    KOpenCLInterface::GetInstance()->GetQueue().finish();
    #endif

    //create the node id list buffer
    CL_ERROR_TRY
    {
        fNodeIDBufferCL
        = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, fTotalSpatialSize*sizeof(unsigned int));
    }
    CL_ERROR_CATCH

    //create the block set id list buffer
    CL_ERROR_TRY
    {
        fBlockSetIDListBufferCL
        = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, fTotalSpatialSize*sizeof(unsigned int));
    }
    CL_ERROR_CATCH

}

////////////////////////////////////////////////////////////////////////////////

void
KFMElectrostaticRemoteToRemoteConverter_OpenCL::AssignBuffers()
{

//ElectrostaticRemoteToRemoteCopyAndScale(const unsigned int n_moment_sets,
//                                        const unsigned int term_stride,
//                                        const unsigned int tree_level,
//                                        __constant const CL_TYPE* scale_factor_array,
//                                        __global unsigned int* node_ids,
//                                        __global unsigned int* block_set_ids,
//                                        __global CL_TYPE2* node_moments,
//                                        __global CL_TYPE2* block_moments)


    fCopyAndScaleKernel->setArg(0, 0); //must be reset before running the kernel
    fCopyAndScaleKernel->setArg(1, fStride);
    fCopyAndScaleKernel->setArg(2, 0); //must be reset before running the kernel
    fCopyAndScaleKernel->setArg(3, *fSourceScaleFactorBufferCL);
    fCopyAndScaleKernel->setArg(4, *fNodeIDBufferCL);
    fCopyAndScaleKernel->setArg(5, *fBlockSetIDListBufferCL);
    fCopyAndScaleKernel->setArg(6, *fNodeMomentBufferCL);
    fCopyAndScaleKernel->setArg(7, *fChildMomentBufferCL);

//ElectrostaticRemoteToRemoteTransform(const unsigned int n_moment_sets,
//                                     const unsigned int degree,
//                                     const unsigned int spatial_stride,
//                                     __global unsigned int* block_set_ids,
//                                     __global CL_TYPE2* response_functions,
//                                     __global CL_TYPE2* original_moments,
//                                     __global CL_TYPE2* transformed_moments)

    fTransformationKernel->setArg(0, 0); //must be reset before running the kernel
    fTransformationKernel->setArg(1, fDegree);
    fTransformationKernel->setArg(2, fTotalSpatialSize);
    fTransformationKernel->setArg(3, *fBlockSetIDListBufferCL);
    fTransformationKernel->setArg(4, *fM2MCoeffBufferCL);
    fTransformationKernel->setArg(5, *fChildMomentBufferCL);
    fTransformationKernel->setArg(6, *fTransformedChildMomentBufferCL);

//ElectrostaticRemoteToRemoteReduceAndScale(const unsigned int n_moment_sets,
//                                          const unsigned int term_stride,
//                                          const unsigned int tree_level,
//                                          const unsigned int parent_node_id,
//                                          __constant const CL_TYPE* scale_factor_array,
//                                          __global unsigned int* block_set_ids,
//                                          __global CL_TYPE2* node_moments,
//                                          __global CL_TYPE2* transformed_child_moments)

    fReduceAndScaleKernel->setArg(0, 0); //must be reset before running the kernel
    fReduceAndScaleKernel->setArg(1, fStride);
    fReduceAndScaleKernel->setArg(2, 0); //must be reset before running the kernel
    fReduceAndScaleKernel->setArg(3, 0); //must be reset before running the kernel
    fReduceAndScaleKernel->setArg(4, *fTargetScaleFactorBufferCL);
    fReduceAndScaleKernel->setArg(5, *fBlockSetIDListBufferCL);
    fReduceAndScaleKernel->setArg(6, *fNodeMomentBufferCL);
    fReduceAndScaleKernel->setArg(7, *fTransformedChildMomentBufferCL);
}

////////////////////////////////////////////////////////////////////////////////

void
KFMElectrostaticRemoteToRemoteConverter_OpenCL::ConstructCopyAndScaleKernel()
{
    //Get name of kernel source file
    std::stringstream clFile;
    clFile << KOpenCLInterface::GetInstance()->GetKernelPath() << "/kEMField_KFMElectrostaticRemoteToRemoteCopyAndScale_kernel.cl";

    KOpenCLKernelBuilder k_builder;
    fCopyAndScaleKernel = k_builder.BuildKernel(clFile.str(), std::string("ElectrostaticRemoteToRemoteCopyAndScale") );

    //get n-local
    fNCopyAndScaleLocal = fCopyAndScaleKernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(KOpenCLInterface::GetInstance()->GetDevice());

    unsigned int preferredWorkgroupMultiple = fCopyAndScaleKernel->getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(KOpenCLInterface::GetInstance()->GetDevice() );

    if(preferredWorkgroupMultiple < fNCopyAndScaleLocal)
    {
        fNCopyAndScaleLocal = preferredWorkgroupMultiple;
    }

}

void
KFMElectrostaticRemoteToRemoteConverter_OpenCL::ConstructTransformationKernel()
{
    //Get name of kernel source file
    std::stringstream clFile;
    clFile << KOpenCLInterface::GetInstance()->GetKernelPath() << "/kEMField_KFMElectrostaticRemoteToRemoteTransformation_kernel.cl";

    KOpenCLKernelBuilder k_builder;
    fTransformationKernel = k_builder.BuildKernel(clFile.str(), std::string("ElectrostaticRemoteToRemoteTransform") );

    //get n-local
    fNTransformationLocal = fTransformationKernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(KOpenCLInterface::GetInstance()->GetDevice());

    unsigned int preferredWorkgroupMultiple = fTransformationKernel->getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(KOpenCLInterface::GetInstance()->GetDevice() );

    if(preferredWorkgroupMultiple < fNTransformationLocal)
    {
        fNTransformationLocal = preferredWorkgroupMultiple;
    }

}

////////////////////////////////////////////////////////////////////////////////

void
KFMElectrostaticRemoteToRemoteConverter_OpenCL::ConstructReduceAndScaleKernel()
{
    //Get name of kernel source file
    std::stringstream clFile;
    clFile << KOpenCLInterface::GetInstance()->GetKernelPath() << "/kEMField_KFMElectrostaticRemoteToRemoteReduceAndScale_kernel.cl";

    KOpenCLKernelBuilder k_builder;
    fReduceAndScaleKernel = k_builder.BuildKernel(clFile.str(), std::string("ElectrostaticRemoteToRemoteReduceAndScale") );

    //get n-local
    fNReduceAndScaleLocal = fReduceAndScaleKernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(KOpenCLInterface::GetInstance()->GetDevice());

    unsigned int preferredWorkgroupMultiple = fReduceAndScaleKernel->getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(KOpenCLInterface::GetInstance()->GetDevice() );

    if(preferredWorkgroupMultiple < fNReduceAndScaleLocal)
    {
        fNReduceAndScaleLocal = preferredWorkgroupMultiple;
    }
}

}
