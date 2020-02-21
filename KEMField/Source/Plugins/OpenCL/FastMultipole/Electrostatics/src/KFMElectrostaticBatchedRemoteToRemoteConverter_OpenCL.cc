#include "KFMElectrostaticBatchedRemoteToRemoteConverter_OpenCL.hh"

namespace KEMField
{



KFMElectrostaticBatchedRemoteToRemoteConverter_OpenCL::KFMElectrostaticBatchedRemoteToRemoteConverter_OpenCL()
{
    fTree = NULL;

    fTransformationKernel = NULL;
    fNTransformationLocal = 0;

    fReduceKernel = NULL;
    fNReduceLocal = 0;

    fM2MCoeffBufferCL = NULL;
    fTransformedMomentBufferCL = NULL;
    fSourceScaleFactorBufferCL = NULL;
    fTargetScaleFactorBufferCL = NULL;
    fNodeMomentBufferCL = NULL;
    fNodeIDBufferCL = NULL;
    fBlockSetIDListBufferCL = NULL;
    fParentNodeOffsetBufferCL = NULL;
    fNChildNodeBufferCL = NULL;
    fParentNodeIDBufferCL = NULL;

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

    fCachedNodeLevel = 0;
    fNMaxParentNodes = 100;
    fNMaxBufferedNodes = 0; //this is set to fNMaxParentNodes*n_children
    fNBufferedNodes = 0;
    fNBufferedParentNodes = 0;

    fKernelResponse = new KFMKernelResponseArray_3DLaplaceM2M(); //false -> origin is the target
    fScaleInvariantKernel = NULL;

    fM2MCoeff = NULL;
};


KFMElectrostaticBatchedRemoteToRemoteConverter_OpenCL::~KFMElectrostaticBatchedRemoteToRemoteConverter_OpenCL()
{
    delete fTransformationKernel;
    delete fReduceKernel;

    delete fM2MCoeffBufferCL;
    delete fTransformedMomentBufferCL;
    delete fSourceScaleFactorBufferCL;
    delete fTargetScaleFactorBufferCL;
    delete fNodeIDBufferCL;
    delete fBlockSetIDListBufferCL;
    delete fParentNodeOffsetBufferCL;
    delete fNChildNodeBufferCL;
    delete fParentNodeIDBufferCL;

    delete fKernelResponse;
    delete fM2MCoeff;


};

void
KFMElectrostaticBatchedRemoteToRemoteConverter_OpenCL::SetParameters(KFMElectrostaticParameters params)
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
KFMElectrostaticBatchedRemoteToRemoteConverter_OpenCL::SetTree(KFMElectrostaticTree* tree)
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
KFMElectrostaticBatchedRemoteToRemoteConverter_OpenCL::SetMultipoleNodeSet(KFMSpecialNodeSet<KFMElectrostaticNodeObjects>* multipole_node_set)
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
KFMElectrostaticBatchedRemoteToRemoteConverter_OpenCL::Initialize()
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


    ConstructTransformationKernel();
    ConstructReduceKernel();

    //n parent nodes should be largest multiple of the local workgroup size
    unsigned int nDummy = fNTransformationLocal - (fNMaxParentNodes%fNTransformationLocal);
    if(nDummy == fNTransformationLocal){nDummy = 0;};
    fNMaxParentNodes += nDummy;

    unsigned int n_children = std::pow(fDivisions, KFMELECTROSTATICS_DIM);
    fNMaxBufferedNodes = fNMaxParentNodes*n_children;

    BuildBuffers();
    AssignBuffers();
}

////////////////////////////////////////////////////////////////////////////////

void
KFMElectrostaticBatchedRemoteToRemoteConverter_OpenCL::CopyMomentsToDevice()
{
    //send the node multipole moments to the device
    unsigned int size = fNMultipoleNodes*fStride;

    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wignored-attributes"
    std::vector< std::complex<CL_TYPE> > moments;
    #pragma GCC diagnostic pop
    moments.resize(size);

    //now distribute the primary node moments
    for(unsigned int i=0; i<fNMultipoleNodes; i++)
    {
        KFMNode<KFMElectrostaticNodeObjects>* node = fMultipoleNodes->GetNodeFromSpecializedID(i);
        KFMElectrostaticMultipoleSet* set = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMElectrostaticMultipoleSet>::GetNodeObject(node);

        if(set != NULL)
        {
            #pragma GCC diagnostic push
            #pragma GCC diagnostic ignored "-Wignored-attributes"
            std::complex<CL_TYPE> temp;
            #pragma GCC diagnostic pop
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
}

////////////////////////////////////////////////////////////////////////////////

void
KFMElectrostaticBatchedRemoteToRemoteConverter_OpenCL::RecieveMomentsFromDevice()
{
    //read the node multipole coefficients back from the device
    unsigned int size = fNMultipoleNodes*fStride;

    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wignored-attributes"
    std::vector< std::complex<CL_TYPE> > moments;
    #pragma GCC diagnostic pop
    moments.resize(size);

    KOpenCLInterface::GetInstance()->GetQueue().enqueueReadBuffer(*fNodeMomentBufferCL, CL_TRUE, 0, size*sizeof(CL_TYPE2), &(moments[0]) );
    #ifdef ENFORCE_CL_FINISH
    KOpenCLInterface::GetInstance()->GetQueue().finish();
    #endif

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
}

////////////////////////////////////////////////////////////////////////////////


void KFMElectrostaticBatchedRemoteToRemoteConverter_OpenCL::Prepare()
{
    fCachedNodeLevel = 0;
    ClearBuffers();
}

void KFMElectrostaticBatchedRemoteToRemoteConverter_OpenCL::Finalize()
{
    ExecuteBufferedAction();
    ClearBuffers();
}

////////////////////////////////////////////////////////////////////////
void
KFMElectrostaticBatchedRemoteToRemoteConverter_OpenCL::ApplyAction(KFMNode<KFMElectrostaticNodeObjects>* node)
{

    if( node != NULL && node->HasChildren() && node->GetLevel() != 0)
    {
        //check if this node is a member of the non-zero multipole node set
        int special_id = fMultipoleNodes->GetSpecializedIDFromOrdinaryID(node->GetID());

        if(special_id != -1)
        {
            //get the number of children which have non-zero multipole moments
            unsigned int n_moment_sets = fCachedMultipoleNodeLists[special_id].size();

            //this nodes contains children that have non-zero multipole moments
            //if it is at the same tree level, and fits, we add it to the buffer
            int node_level = node->GetLevel();

            if(node_level != fCachedNodeLevel)
            {
                //we have entered a new tree level we need to execute the kernel and clear
                //the buffered nodes, before adding this node to the buffer
                //because the next batch will need to use different scale factors
                ExecuteBufferedAction();
                ClearBuffers();
                BufferNode(node);
                fCachedNodeLevel = node_level;
            }
            else
            {
                //now determine if we can fit the node into the current buffer
                if( (fNBufferedNodes + n_moment_sets < fNMaxBufferedNodes) && (fNBufferedParentNodes < fNMaxParentNodes) )
                {
                    //all we have to do is buffer this node
                    BufferNode(node);
                }
                else
                {
                    //execute action on already buffered nodes
                    ExecuteBufferedAction();
                    ClearBuffers();
                    BufferNode(node);
                }
            }
        }
    }
}


void
KFMElectrostaticBatchedRemoteToRemoteConverter_OpenCL::BufferNode(KFMNode<KFMElectrostaticNodeObjects>* node)
{
    //check if this node is a member of the non-zero multipole node set
    int special_id = fMultipoleNodes->GetSpecializedIDFromOrdinaryID(node->GetID());
    //get the number of children which have non-zero multipole moments
    unsigned int n_moment_sets = fCachedMultipoleNodeLists[special_id].size();

    //fill in the data needed for the reduction kernel
    fParentNodeOffsetBuffer.push_back(fNBufferedNodes);
    fNChildBuffer.push_back(n_moment_sets);
    fParentNodeIDBuffer.push_back(special_id);
    fNBufferedParentNodes++;

    //buffer data for the transformation kernel
    //fill in the node ids, block set ids for all nodes with moments sets
    for(unsigned int i=0; i<n_moment_sets; i++)
    {
        fNodeIDBuffer.push_back(fCachedMultipoleNodeLists[special_id][i]);
        fBlockSetIDBuffer.push_back(fCachedBlockSetIDLists[special_id][i]);
        fNBufferedNodes++;
    }
}

void
KFMElectrostaticBatchedRemoteToRemoteConverter_OpenCL::ExecuteBufferedAction()
{
    if(fNBufferedNodes != 0 && fNBufferedParentNodes != 0)
    {
        unsigned int nDummy;
        unsigned int nLocal;
        unsigned int nGlobal;

        ///////////
        //set up transformation kernel data

        fTransformationKernel->setArg(0, fNBufferedNodes); //n buffered nodes
        fTransformationKernel->setArg(1, fCachedNodeLevel+1); //tree level

        //write out the child node id's
        KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fNodeIDBufferCL, CL_TRUE, 0, fNBufferedNodes*sizeof(unsigned int), &(fNodeIDBuffer[0]));
        #ifdef ENFORCE_CL_FINISH
        KOpenCLInterface::GetInstance()->GetQueue().finish();
        #endif

        //write out this nodes block set id's
        KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fBlockSetIDListBufferCL, CL_TRUE, 0, fNBufferedNodes*sizeof(unsigned int), &(fBlockSetIDBuffer[0]));
        #ifdef ENFORCE_CL_FINISH
        KOpenCLInterface::GetInstance()->GetQueue().finish();
        #endif

        //execute transformation kernel
        nLocal = fNTransformationLocal;
        nGlobal = fNBufferedNodes*fStride;
        nDummy = nLocal - (nGlobal%nLocal);
        if(nDummy == nLocal){nDummy = 0;};
        KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fTransformationKernel, cl::NullRange, cl::NDRange(nGlobal + nDummy), cl::NDRange(nLocal) );
        #ifdef ENFORCE_CL_FINISH
        KOpenCLInterface::GetInstance()->GetQueue().finish();
        #endif

        ////////////
        //set up reduction kernel data

        fReduceKernel->setArg(0, fNBufferedParentNodes); //n buffered parent nodes

        //write out the offset to the parents node data
        KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fParentNodeOffsetBufferCL, CL_TRUE, 0, fNBufferedParentNodes*sizeof(unsigned int), &(fParentNodeOffsetBuffer[0]));
        #ifdef ENFORCE_CL_FINISH
        KOpenCLInterface::GetInstance()->GetQueue().finish();
        #endif

        //write out the parent nodes size of child data
        KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fNChildNodeBufferCL, CL_TRUE, 0, fNBufferedParentNodes*sizeof(unsigned int), &(fNChildBuffer[0]));
        #ifdef ENFORCE_CL_FINISH
        KOpenCLInterface::GetInstance()->GetQueue().finish();
        #endif

        //write out the parent node id data
        KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fParentNodeIDBufferCL, CL_TRUE, 0, fNBufferedParentNodes*sizeof(unsigned int), &(fParentNodeIDBuffer[0]));
        #ifdef ENFORCE_CL_FINISH
        KOpenCLInterface::GetInstance()->GetQueue().finish();
        #endif

        //run the reduce and scale kernel
        nLocal = fNReduceLocal;
        nGlobal = fNBufferedParentNodes*fStride;
        nDummy = nLocal - (nGlobal%nLocal);
        if(nDummy == nLocal){nDummy = 0;};
        KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fReduceKernel, cl::NullRange, cl::NDRange(nGlobal + nDummy), cl::NDRange(nLocal) );
        #ifdef ENFORCE_CL_FINISH
        KOpenCLInterface::GetInstance()->GetQueue().finish();
        #endif

    }
}

void
KFMElectrostaticBatchedRemoteToRemoteConverter_OpenCL::ClearBuffers()
{
        //vectors of offset, size, and ids for reduction kernel
    fParentNodeOffsetBuffer.clear();
    fNChildBuffer.clear();
    fParentNodeIDBuffer.clear();

    fNodeIDBuffer.clear();
    fBlockSetIDBuffer.clear();

    fNBufferedNodes = 0;
    fNBufferedParentNodes = 0;
}



void
KFMElectrostaticBatchedRemoteToRemoteConverter_OpenCL::BuildBuffers()
{

//    fM2MCoeffBufferCL; !!!!
//    fTransformedMomentBufferCL; !!!!
//    fSourceScaleFactorBufferCL; !!!!
//    fTargetScaleFactorBufferCL; !!!!
//    fNodeMomentBufferCL; //not created here
//    fNodeIDBufferCL; !!!!
//    fBlockSetIDListBufferCL; !!!!
//    fParentNodeOffsetBufferCL; !!!!
//    fNChildNodeBufferCL; !!!!
//    fParentNodeIDBufferCL; !!!!

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

    //create the transformed children's moment buffer
    CL_ERROR_TRY
    {
        fTransformedMomentBufferCL
        = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_WRITE, fNMaxBufferedNodes*fStride*sizeof(CL_TYPE2));
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
        = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, fNMaxBufferedNodes*sizeof(unsigned int));
    }
    CL_ERROR_CATCH

    //create the block set id list buffer
    CL_ERROR_TRY
    {
        fBlockSetIDListBufferCL
        = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, fNMaxBufferedNodes*sizeof(unsigned int));
    }
    CL_ERROR_CATCH

    //create the block set id list buffer
    CL_ERROR_TRY
    {
        fParentNodeOffsetBufferCL
        = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, fNMaxParentNodes*sizeof(unsigned int));
    }
    CL_ERROR_CATCH

    CL_ERROR_TRY
    {
        fNChildNodeBufferCL
        = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, fNMaxParentNodes*sizeof(unsigned int));
    }
    CL_ERROR_CATCH

    CL_ERROR_TRY
    {
        fParentNodeIDBufferCL
        = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, fNMaxParentNodes*sizeof(unsigned int));
    }
    CL_ERROR_CATCH

    std::vector<unsigned int> zero_buff;
    zero_buff.resize(fNMaxBufferedNodes, 0);

    //write out zeros to the id buffers
    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fParentNodeOffsetBufferCL, CL_TRUE, 0, fNMaxParentNodes*sizeof(unsigned int), &(zero_buff[0]) );
    #ifdef ENFORCE_CL_FINISH
    KOpenCLInterface::GetInstance()->GetQueue().finish();
    #endif

    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fNChildNodeBufferCL, CL_TRUE, 0, fNMaxParentNodes*sizeof(unsigned int), &(zero_buff[0]) );
    #ifdef ENFORCE_CL_FINISH
    KOpenCLInterface::GetInstance()->GetQueue().finish();
    #endif

    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fParentNodeIDBufferCL, CL_TRUE, 0, fNMaxParentNodes*sizeof(unsigned int), &(zero_buff[0]) );
    #ifdef ENFORCE_CL_FINISH
    KOpenCLInterface::GetInstance()->GetQueue().finish();
    #endif

}

////////////////////////////////////////////////////////////////////////////////

void
KFMElectrostaticBatchedRemoteToRemoteConverter_OpenCL::AssignBuffers()
{

//ElectrostaticBatchedRemoteToRemoteTransform(const unsigned int n_nodes,
//                                            const unsigned int tree_level,
//                                            __constant const CL_TYPE* source_scale_factor_array,
//                                            __constant const CL_TYPE* target_scale_factor_array,
//                                            __global CL_TYPE2* response_functions,
//                                            __global CL_TYPE2* node_moments,
//                                            __global CL_TYPE2* transformed_moments,
//                                            __global unsigned int* node_ids,
//                                            __global unsigned int* block_set_ids)

    fTransformationKernel->setArg(0, 0); //must be set before running the kernel
    fTransformationKernel->setArg(1, 0); //must be set before running the kernel
    fTransformationKernel->setArg(2, *fSourceScaleFactorBufferCL);
    fTransformationKernel->setArg(3, *fTargetScaleFactorBufferCL);
    fTransformationKernel->setArg(4, *fM2MCoeffBufferCL);
    fTransformationKernel->setArg(5, *fNodeMomentBufferCL);
    fTransformationKernel->setArg(6, *fTransformedMomentBufferCL);
    fTransformationKernel->setArg(7, *fNodeIDBufferCL); //must write before running kernel
    fTransformationKernel->setArg(8, *fBlockSetIDListBufferCL); //must write before running kernel

//ElectrostaticBatchedRemoteToRemoteReduce(const unsigned int n_parent_nodes,
//                                         __global const unsigned int* parent_node_offset,
//                                         __global const unsigned int* n_child_nodes,
//                                         __global const unsigned int* parent_node_ids,
//                                         __global CL_TYPE2* node_moments,
//                                         __global CL_TYPE2* transformed_moments)

    fReduceKernel->setArg(0, 0); //must be reset before running the kernel
    fReduceKernel->setArg(1, *fParentNodeOffsetBufferCL); //must write before running kernel
    fReduceKernel->setArg(2, *fNChildNodeBufferCL); //must write before running kernel
    fReduceKernel->setArg(3, *fParentNodeIDBufferCL); //must write before running kernel
    fReduceKernel->setArg(4, *fNodeMomentBufferCL);
    fReduceKernel->setArg(5, *fTransformedMomentBufferCL);

}

////////////////////////////////////////////////////////////////////////////////

void
KFMElectrostaticBatchedRemoteToRemoteConverter_OpenCL::ConstructTransformationKernel()
{
    //Get name of kernel source file
    std::stringstream clFile;
    clFile << KOpenCLInterface::GetInstance()->GetKernelPath() << "/kEMField_KFMElectrostaticBatchedRemoteToRemoteTransformation_kernel.cl";

    //create the build flags
    std::stringstream ss;
    ss << " -D KFM_DEGREE=" <<fDegree;
    ss << " -D KFM_REAL_STRIDE=" << fStride;
    ss << " -D KFM_COMPLEX_STRIDE=" << (fDegree+1)*(fDegree+1);
    ss << " -D KFM_SPATIAL_STRIDE=" << fTotalSpatialSize;

    std::string build_flags = ss.str();

    KOpenCLKernelBuilder k_builder;
    fTransformationKernel = k_builder.BuildKernel(clFile.str(), std::string("ElectrostaticBatchedRemoteToRemoteTransform"), build_flags );

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
KFMElectrostaticBatchedRemoteToRemoteConverter_OpenCL::ConstructReduceKernel()
{
    //Get name of kernel source file
    std::stringstream clFile;
    clFile << KOpenCLInterface::GetInstance()->GetKernelPath() << "/kEMField_KFMElectrostaticBatchedRemoteToRemoteReduce_kernel.cl";

    //create the build flags
    std::stringstream ss;
    ss << " -D KFM_DEGREE=" <<fDegree;
    ss << " -D KFM_REAL_STRIDE=" << fStride;
    ss << " -D KFM_COMPLEX_STRIDE=" << (fDegree+1)*(fDegree+1);
    ss << " -D KFM_SPATIAL_STRIDE=" << fTotalSpatialSize;

    std::string build_flags = ss.str();

    KOpenCLKernelBuilder k_builder;
    fReduceKernel = k_builder.BuildKernel(clFile.str(), std::string("ElectrostaticBatchedRemoteToRemoteReduce"), build_flags );

    //get n-local
    fNReduceLocal = fReduceKernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(KOpenCLInterface::GetInstance()->GetDevice());

    unsigned int preferredWorkgroupMultiple = fReduceKernel->getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(KOpenCLInterface::GetInstance()->GetDevice() );

    if(preferredWorkgroupMultiple < fNReduceLocal)
    {
        fNReduceLocal = preferredWorkgroupMultiple;
    }
}

}
