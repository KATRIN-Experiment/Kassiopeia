#include "KFMElectrostaticRemoteToLocalConverter_OpenCL.hh"

namespace KEMField
{



KFMElectrostaticRemoteToLocalConverter_OpenCL::KFMElectrostaticRemoteToLocalConverter_OpenCL()
{
    fTree = NULL;

    fDegree = 0;
    fNTerms = 0;
    fStride = 0;
    fNResponseTerms = 0;
    fDivisions = 0;
    fTopLevelDivisions = 0;
    fLowerLevelDivisions = 0;
    fZeroMaskSize = 0;
    fNeighborOrder = 0;
    fMaxTreeDepth = 0;
    fDim = 0;
    fTotalSpatialSize = 0;
    fNeighborStride = 0;
    fWorldLength = 0;
    fFFTNormalization = 1.0;

    fKernelResponse = new KFMKernelReducedResponseArray_3DLaplaceM2L();
    fScaleInvariantKernel = dynamic_cast< KFMScaleInvariantKernelExpansion<KFMELECTROSTATICS_DIM>* >( fKernelResponse->GetKernel() );

    fHelperArrayWrapper = NULL;

    fAllM2LCoeff = NULL;
    fDFTCalc = new KFMMultidimensionalFastFourierTransform< KFMELECTROSTATICS_DIM >();

    fZeroComplexArrayKernel = NULL;
    fNZeroComplexArrayLocal = 0;

    fCopyAndScaleKernel = NULL;
    fNCopyAndScaleLocal = 0;

    fTransformationKernel = NULL;
    fNTransformationLocal = 0;

    fReduceAndScaleKernel = NULL;
    fNReduceAndScaleLocal = 0;

    fDFTCalcOpenCL = new KFMBatchedMultidimensionalFastFourierTransform_OpenCL<KFMELECTROSTATICS_DIM>();

    fM2LCoeffBufferCL = NULL;
    fFFTDataBufferCL = NULL;
    fWorkspaceBufferCL = NULL;
    fNormalizationCoeffBufferCL = NULL;
    fReversedIndexArrayBufferCL = NULL;

    fNMultipoleNodes = 0;
    fNPrimaryNodes = 0;
    fMultipoleNodes = NULL;
    fPrimaryNodes = NULL;

    fNodeLocalMomentBufferCL = NULL;
    fNodeRemoteMomentBufferCL = NULL;

    fMultipoleNodeIDListBufferCL = NULL;
    fMultipoleBlockSetIDListBufferCL = NULL;
    fPrimaryNodeIDListBufferCL = NULL;
    fPrimaryBlockSetIDListBufferCL = NULL;
}

KFMElectrostaticRemoteToLocalConverter_OpenCL::~KFMElectrostaticRemoteToLocalConverter_OpenCL()
{
    delete fKernelResponse;

    delete fHelperArrayWrapper;
    delete fAllM2LCoeff;
    delete fDFTCalc;

    delete fZeroComplexArrayKernel;
    delete fCopyAndScaleKernel;
    delete fTransformationKernel;
    delete fReduceAndScaleKernel;

    delete fDFTCalcOpenCL;

    delete fM2LCoeffBufferCL;
    delete fWorkspaceBufferCL;
    delete fNormalizationCoeffBufferCL;
    delete fReversedIndexArrayBufferCL;

    delete fMultipoleNodeIDListBufferCL;
    delete fMultipoleBlockSetIDListBufferCL;
    delete fPrimaryNodeIDListBufferCL;
    delete fPrimaryBlockSetIDListBufferCL;
}

//void
//KFMElectrostaticRemoteToLocalConverter_OpenCL::SetParameters(KFMElectrostaticParameters params)
//{
//    //set parameters
//    fDegree = params.degree;
//    fNTerms = (fDegree+1)*(fDegree+1);
//    fStride = (fDegree+1)*(fDegree+2)/2;
//    fNResponseTerms = (2*fDegree+1)*(2*fDegree+1);
//    fDivisions = params.divisions;
//    fMaxTreeDepth = params.maximum_tree_depth;
//    fZeroMaskSize = params.zeromask;
//    fDim = 2*fDivisions*(fZeroMaskSize + 1);
//    fNeighborStride = 2*fZeroMaskSize + 1;
//}


void
KFMElectrostaticRemoteToLocalConverter_OpenCL::SetTree(KFMElectrostaticTree* tree)
{
    fTree = tree;

//    //determine world region size to compute scale factors
//    KFMCube<KFMELECTROSTATICS_DIM>* world_cube =
//    KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCube<KFMELECTROSTATICS_DIM> >::GetNodeObject(fTree->GetRootNode());
//    fWorldLength = world_cube->GetLength();

    //now we want to retrieve the top level and lower level divisions
    //since we need both in order to compute the scale factor for the different tree levels correctly
    const unsigned int* dim_size;
    dim_size = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCubicSpaceTreeProperties<KFMELECTROSTATICS_DIM> >::GetNodeObject(fTree->GetRootNode())->GetTopLevelDimensions();
    fTopLevelDivisions = dim_size[0];

    dim_size = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCubicSpaceTreeProperties<KFMELECTROSTATICS_DIM> >::GetNodeObject(fTree->GetRootNode())->GetDimensions();
    fLowerLevelDivisions = dim_size[0];
}

void
KFMElectrostaticRemoteToLocalConverter_OpenCL::SetMultipoleNodeSet(KFMSpecialNodeSet<KFMElectrostaticNodeObjects>* multipole_node_set)
{
    fMultipoleNodes = multipole_node_set;
    fNMultipoleNodes = fMultipoleNodes->GetSize();
}

void
KFMElectrostaticRemoteToLocalConverter_OpenCL::SetPrimaryNodeSet(KFMSpecialNodeSet<KFMElectrostaticNodeObjects>* local_node_set)
{
    fPrimaryNodes = local_node_set;
    fNPrimaryNodes = fPrimaryNodes->GetSize();
}

void
KFMElectrostaticRemoteToLocalConverter_OpenCL::ConstructCachedNodeIdentityLists()
{
    fCachedMultipoleNodeIDLists.resize(fNMultipoleNodes);
    fCachedMultipoleBlockSetIDLists.resize(fNMultipoleNodes);
    fCachedPrimaryNodeIDLists.resize(fNMultipoleNodes);
    fCachedPrimaryBlockSetIDLists.resize(fNMultipoleNodes);

    fMultipoleNodeIDList.clear();
    fMultipoleBlockSetIDList.clear();
    fPrimaryNodeIDList.clear();
    fPrimaryBlockSetIDList.clear();

    fMultipoleNodeIDListStartIndexes.clear();
    fMultipoleBlockSetIDListStartIndexes.clear();
    fPrimaryNodeIDListStartIndexes.clear();
    fPrimaryBlockSetIDListStartIndexes.clear();

    unsigned int n_multipole_ids = 0;

    // unsigned int dim_size[KFMELECTROSTATICS_DIM];
    // for(unsigned int i=0; i<KFMELECTROSTATICS_DIM; i++){dim_size[i] = fDivisions;};

    //fill the list of child nodes with multipole moments and their positions (block set ids)
    for(unsigned int i=0; i<fNMultipoleNodes; i++)
    {
        fCachedMultipoleNodeIDLists[i].clear();
        fCachedMultipoleBlockSetIDLists[i].clear();

        fMultipoleNodeIDListStartIndexes.push_back(fMultipoleNodeIDList.size());
        fMultipoleBlockSetIDListStartIndexes.push_back(fMultipoleBlockSetIDList.size());

        KFMNode<KFMElectrostaticNodeObjects>* node = fMultipoleNodes->GetNodeFromSpecializedID(i);

       //division dimensions
       const unsigned int* dim_size;
       if(node->GetLevel() == 0)
       {
           dim_size =
           KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCubicSpaceTreeProperties<KFMELECTROSTATICS_DIM> >::GetNodeObject(node)->GetTopLevelDimensions();
       }
       else
       {
           dim_size =
       KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCubicSpaceTreeProperties<KFMELECTROSTATICS_DIM> >::GetNodeObject(node)->GetDimensions();
       }

        int index[KFMELECTROSTATICS_DIM + 1];
        unsigned int spatial_index[KFMELECTROSTATICS_DIM];

        if(node->HasChildren())
        {
            unsigned int n_children = node->GetNChildren();
            for(unsigned int j=0; j<n_children; j++)
            {
                int special_id = fMultipoleNodes->GetSpecializedIDFromOrdinaryID(node->GetChild(j)->GetID());

                if(special_id != -1)
                {
                    unsigned int id = static_cast<unsigned int>(special_id);
                    fCachedMultipoleNodeIDLists[i].push_back(id);
                    fMultipoleNodeIDList.push_back(id);

                    //now we compute this child's block set id
                    KFMArrayMath::RowMajorIndexFromOffset<KFMELECTROSTATICS_DIM>(j, dim_size, spatial_index);

                    index[0] = 0;
                    for(unsigned int n=0; n<KFMELECTROSTATICS_DIM; n++)
                    {
                        index[n+1] = spatial_index[n];
                    }

                    unsigned int block_id = fHelperArrayWrapper->GetOffsetForIndices(index);
                    fCachedMultipoleBlockSetIDLists[i].push_back(block_id);
                    fMultipoleBlockSetIDList.push_back(block_id);

                    n_multipole_ids++;
                }
            }
        }
    }

    //temporary variables to do collections
    KFMNode<KFMElectrostaticNodeObjects>* child;
    std::vector< KFMNode<KFMElectrostaticNodeObjects>* > neighbors;

    unsigned int n_primary_ids = 0;

    //fill the list of primary nodes that are associated with each node that has children with non-zero multipole moments
    for(unsigned int i=0; i<fNMultipoleNodes; i++)
    {
        fCachedPrimaryNodeIDLists[i].clear();
        fCachedPrimaryBlockSetIDLists[i].clear();


        fPrimaryNodeIDListStartIndexes.push_back(fPrimaryNodeIDList.size());
        fPrimaryBlockSetIDListStartIndexes.push_back(fPrimaryBlockSetIDList.size());

        KFMNode<KFMElectrostaticNodeObjects>* node = fMultipoleNodes->GetNodeFromSpecializedID(i);

        if(node->HasChildren())
        {
            unsigned int szpn[KFMELECTROSTATICS_DIM]; //parent neighbor spatial index
            unsigned int sznc[KFMELECTROSTATICS_DIM]; //neighbor child spatial index (within neighbor)

            int pn[KFMELECTROSTATICS_DIM]; //parent neighbor spatial index (relative position to original node)
            int lc[KFMELECTROSTATICS_DIM]; //global position in local coefficient array of this child

            unsigned int offset; //offset due to spatial indices from beginning of local coefficient array of this child

            //get all neighbors of this node
            KFMCubicSpaceNodeNeighborFinder<KFMELECTROSTATICS_DIM, KFMElectrostaticNodeObjects>::GetAllNeighbors(node, fNeighborOrder, &(neighbors));

            for(unsigned int n=0; n < neighbors.size(); n++)
            {
                if(neighbors[n] != NULL)
                {
                    //compute relative index of this neighbor and store in pn array
                    KFMArrayMath::RowMajorIndexFromOffset<KFMELECTROSTATICS_DIM>(n, fNeighborDimensionSize, szpn);
                    for(unsigned int x=0; x<KFMELECTROSTATICS_DIM; x++)
                    {
                        pn[x] = (int)szpn[x] - fNeighborOrder;
                    }

                    //loop over neighbors children
                    unsigned int n_children = neighbors[n]->GetNChildren();

                   //division dimensions
                   const unsigned int* dim_size;
                   if(neighbors[n]->GetLevel() == 0)
                   {
                       dim_size =
                       KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCubicSpaceTreeProperties<KFMELECTROSTATICS_DIM> >::GetNodeObject(neighbors[n])->GetTopLevelDimensions();
                   }
                   else
                   {
                       dim_size =
                   KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCubicSpaceTreeProperties<KFMELECTROSTATICS_DIM> >::GetNodeObject(neighbors[n])->GetDimensions();
                   }

                    for(unsigned int c = 0; c < n_children; c++)
                    {
                        child = neighbors[n]->GetChild(c);
                        if(child != NULL)
                        {
                            //get child's id
                            unsigned int child_id = child->GetID();

                            //look up if this child is a primary node
                            int child_primary_node_id = fPrimaryNodes->GetSpecializedIDFromOrdinaryID(child_id);

                            if(child_primary_node_id != -1)
                            {
                                //we have a primary node, write it's primary id to the list
                                KFMArrayMath::RowMajorIndexFromOffset<KFMELECTROSTATICS_DIM>(c, dim_size, sznc);

                                //spatial index of local coefficients for this child
                                for(unsigned int x=0; x<KFMELECTROSTATICS_DIM; x++)
                                {
                                    lc[x] = (pn[x])*(fDivisions) + (int)sznc[x];
                                }

                                offset = fM2LCoeff[0]->GetOffsetForIndices(lc);

                                fCachedPrimaryNodeIDLists[i].push_back(static_cast<unsigned int>(child_primary_node_id));
                                fPrimaryNodeIDList.push_back(static_cast<unsigned int>(child_primary_node_id));

                                fCachedPrimaryBlockSetIDLists[i].push_back(offset);
                                fPrimaryBlockSetIDList.push_back(offset);

                                n_primary_ids++;
                            }
                        }
                    }
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

void KFMElectrostaticRemoteToLocalConverter_OpenCL::Prepare()
{
    //zero out the local coefficient array

    unsigned int n_global;
    unsigned int nDummy;

    //first we have to zero out the local coeff buffer
    fZeroComplexArrayKernel->setArg(0, fStride*fNPrimaryNodes);
    fZeroComplexArrayKernel->setArg(1, *fNodeLocalMomentBufferCL);

    //compute size of the array
    n_global = fStride*fNPrimaryNodes;

    nDummy = fNZeroComplexArrayLocal - (n_global%fNZeroComplexArrayLocal);
    if(nDummy == fNZeroComplexArrayLocal){nDummy = 0;};
    n_global += nDummy;

    //now enqueue the kernel
    cl::Event zero_event;
    KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fZeroComplexArrayKernel, cl::NullRange, cl::NDRange(n_global), cl::NDRange(fNZeroComplexArrayLocal), NULL, &zero_event);
    zero_event.wait();

    #ifdef ENFORCE_CL_FINISH
    KOpenCLInterface::GetInstance()->GetQueue().finish();
    #endif

}

////////////////////////////////////////////////////////////////////////////////

//this function is called after visiting the tree to finalize the tree state if needed
void
KFMElectrostaticRemoteToLocalConverter_OpenCL::Finalize()
{
    // //read the primary node local coefficients back from the gpu;
    // unsigned int primary_size = fNPrimaryNodes*fStride;
    // fPrimaryLocalCoeff.resize(fStride*fNPrimaryNodes);
    //
    // KOpenCLInterface::GetInstance()->GetQueue().enqueueReadBuffer(*fNodeLocalMomentBufferCL, CL_TRUE, 0, primary_size*sizeof(CL_TYPE2), &(fPrimaryLocalCoeff[0]) );
    //
    // #ifdef ENFORCE_CL_FINISH
    // KOpenCLInterface::GetInstance()->GetQueue().finish();
    // #endif
    //
    // //now distribute the primary node moments
    // for(unsigned int i=0; i<fNPrimaryNodes; i++)
    // {
    //     KFMNode<KFMElectrostaticNodeObjects>* node = fPrimaryNodes->GetNodeFromSpecializedID(i);
    //     KFMElectrostaticLocalCoefficientSet* set = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMElectrostaticLocalCoefficientSet>::GetNodeObject(node);
    //
    //     if(set != NULL)
    //     {
    //         std::complex<double> temp;
    //         //we use raw ptr for speed
    //         double* rmoments = &( (*(set->GetRealMoments()))[0] );
    //         double* imoments = &( (*(set->GetImaginaryMoments()))[0] );
    //         for(unsigned int j=0; j < fStride; ++j)
    //         {
    //             temp = fPrimaryLocalCoeff[i*fStride + j];
    //             rmoments[j] = temp.real();
    //             imoments[j] = temp.imag();
    //         }
    //     }
    // }
    //
    // fPrimaryLocalCoeff.resize(0);

};

////////////////////////////////////////////////////////////////////////////////


void
KFMElectrostaticRemoteToLocalConverter_OpenCL::Initialize()
{

    for(unsigned int i=0; i<KFMELECTROSTATICS_DIM; i++)
    {
        fNeighborDimensionSize[i] = fNeighborStride;
    }

    //set number of terms in series for response functions
    fKernelResponse->SetNumberOfTermsInSeries(fNResponseTerms);
    fKernelResponse->SetZeroMaskSize(fZeroMaskSize);

    //set dimensions limits
    fTargetLowerLimits[0] = 0;
    fTargetUpperLimits[0] = fStride;
    fTargetDimensionSize[0] = fStride;

    fSourceLowerLimits[0] = 0;
    fSourceUpperLimits[0] = fStride;
    fSourceDimensionSize[0] = fStride;

    fLowerResponseLimits[0] = 0;
    fUpperResponseLimits[0] = fNResponseTerms;
    fResponseDimensionSize[0] = fNResponseTerms;

    for(unsigned int i=0; i<KFMELECTROSTATICS_DIM; i++)
    {
        fTargetLowerLimits[i+1] = -1*(fNeighborOrder+1)*fDivisions;
        fTargetUpperLimits[i+1] = fTargetLowerLimits[i+1] + fDim;
        fTargetDimensionSize[i+1] =  fDim;

        fSourceLowerLimits[i+1] = -1*(fNeighborOrder+1)*fDivisions;
        fSourceUpperLimits[i+1] = fSourceLowerLimits[i+1] + fDim;
        fSourceDimensionSize[i+1] =  fDim;

        fChildDimensionSize[i] = fDivisions;
    }

    for(unsigned int i=0; i<KFMELECTROSTATICS_DIM; i++)
    {
        fLowerResponseLimits[i+1] = -1*(fNeighborOrder+1)*fDivisions;
        fUpperResponseLimits[i+1] = (fNeighborOrder+1)*fDivisions;
        fResponseDimensionSize[i+1] = fDim;
    }

    fTotalSpatialSize = KFMArrayMath::TotalArraySize<KFMELECTROSTATICS_DIM>( &(fTargetDimensionSize[1]) );

    /////////////////////
    //helper array wrapper
    fRawHelperArray.resize(fStride*fTotalSpatialSize);
    fHelperArrayWrapper = new KFMArrayWrapper<std::complex<double>, KFMELECTROSTATICS_DIM + 1>( &(fRawHelperArray[0]), fSourceDimensionSize );

    //initialize the opencl dft batch calculator
    //intialize DFT calculator for array dimensions
    fDFTCalcOpenCL->SetInput(fHelperArrayWrapper);
    fDFTCalcOpenCL->SetOutput(fHelperArrayWrapper);

    //all enqueue read/write buffers occur external to the DFT kernel execution
    fDFTCalcOpenCL->SetWriteOutHostDataFalse(); //all data already on device
    fDFTCalcOpenCL->SetReadOutDataToHostFalse(); //all data stays on device
    fDFTCalcOpenCL->Initialize();

    //swap raw helper array with empty vector, since it is no longer used
    std::vector< std::complex<double> > temp;
    temp.swap(fRawHelperArray);

    /////////////////////

    fKernelResponse->SetLowerSpatialLimits(&(fLowerResponseLimits[1]));
    fKernelResponse->SetUpperSpatialLimits(&(fUpperResponseLimits[1]));

    fFFTNormalization =  std::pow( (double)(fTotalSpatialSize), -1.);

    //allocate space and wrappers for M2L coeff
    fRawM2LCoeff.resize(fNResponseTerms*fTotalSpatialSize);
    fAllM2LCoeff = new KFMArrayWrapper<std::complex<double>, KFMELECTROSTATICS_DIM + 1>( &(fRawM2LCoeff[0]), fResponseDimensionSize);
    fAllM2LCoeff->SetArrayBases(fLowerResponseLimits);
    fM2LCoeff.resize(fNResponseTerms, NULL);

    std::complex<double>* ptr;
    for(unsigned int tsi=0; tsi<fNResponseTerms; tsi++)
    {
        ptr = &(fRawM2LCoeff[tsi*fTotalSpatialSize] );
        fM2LCoeff[tsi] = new KFMArrayWrapper<std::complex<double>, KFMELECTROSTATICS_DIM >(ptr,  &(fResponseDimensionSize[1]) );
        fM2LCoeff[tsi]->SetArrayBases(&(fLowerResponseLimits[1]));
    }


    ////////////////////////////////////////////////////////////////////////////
    //now compute M2L coefficients

    fKernelResponse->SetDistance(1.0);
    fKernelResponse->SetOutput(fAllM2LCoeff);
    fKernelResponse->Initialize();
    fKernelResponse->ExecuteOperation();


    //now we have to perform the dft on all the M2L coefficients
    fDFTCalc->SetForward();
    for(unsigned int tsi=0; tsi<fNResponseTerms; tsi++)
    {
        //dft calc must be initialized with arrays of the same size
        //before being used here
        fDFTCalc->SetInput(fM2LCoeff[tsi]);
        fDFTCalc->SetOutput(fM2LCoeff[tsi]);
        fDFTCalc->Initialize();
        fDFTCalc->ExecuteOperation();
    }

    //allocate and compute normalization factors
    fNormalizationCoeff.resize(fNTerms*fNTerms);
    for(unsigned int tsi=0; tsi<fNTerms; tsi++)
    {
        for(unsigned int ssi=0; ssi<fNTerms; ssi++)
        {
            fNormalizationCoeff[ssi + tsi*fNTerms] = fFFTNormalization*(fKernelResponse->GetKernel()->GetNormalizationFactor(ssi, tsi));
        }
    }

    ////////////////////////////////////////////////////////////////////////////

    //now allocate compute the source and target scale factors
    fSourceScaleFactorArray.resize((fMaxTreeDepth+1)*fStride);
    fTargetScaleFactorArray.resize((fMaxTreeDepth+1)*fStride);

    double level_side_length = fWorldLength;
    double div_power;

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

    CheckDeviceProperites(); //make sure device has enough space

    ConstructCachedNodeIdentityLists();//collect all the node/block set id lists

    //fill reversed array look up table
    fReversedIndexArray.resize(fTotalSpatialSize);
    KFMArrayMath::OffsetsForReversedIndices<KFMELECTROSTATICS_DIM>( &(fTargetDimensionSize[1]), &(fReversedIndexArray[0]) );

    ConstructZeroComplexArrayKernel();
    ConstructCopyAndScaleKernel();
    ConstructTransformationKernel();
    ConstructReduceAndScaleKernel();

    BuildBuffers();

    AssignBuffers();


}

////////////////////////////////////////////////////////////////////////////////

void
KFMElectrostaticRemoteToLocalConverter_OpenCL::ApplyAction(KFMNode<KFMElectrostaticNodeObjects>* node)
{
    if( node != NULL && node->HasChildren() )
    {
        //check if this node is a member of the non-zero multipole node set
        int special_id = fMultipoleNodes->GetSpecializedIDFromOrdinaryID(node->GetID());

        if(special_id != -1)
        {
            //get the number of children which have non-zero multipole moments
            //and et the number of moment sets argument
            unsigned int n_multipole_sets = fCachedMultipoleNodeIDLists[special_id].size();
            unsigned int n_primary_sets = fCachedPrimaryNodeIDLists[special_id].size();
            unsigned int tree_level = node->GetLevel() + 1;

            if(n_multipole_sets != 0)
            {
                //set kernel args and update buffers ---------------------------
                fCopyAndScaleKernel->setArg(0, n_multipole_sets);
                fReduceAndScaleKernel->setArg(0, n_primary_sets);

                //tree level
                fCopyAndScaleKernel->setArg(3, tree_level);
                fReduceAndScaleKernel->setArg(3, tree_level);

                fCopyAndScaleKernel->setArg(4, fMultipoleBlockSetIDListStartIndexes[special_id]);

                fReduceAndScaleKernel->setArg(4, fPrimaryBlockSetIDListStartIndexes[special_id]);

                //--------------------------------------------------------------

                ApplyCopyAndScaleKernel(n_multipole_sets);

                ApplyTransformationKernel();

                ApplyReduceAndScaleKernel(n_primary_sets);

            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

void
KFMElectrostaticRemoteToLocalConverter_OpenCL::BuildBuffers()
{
    //create the m2l buffer
    size_t m2l_size = fNResponseTerms*fTotalSpatialSize;
    CL_ERROR_TRY
    {
        fM2LCoeffBufferCL
        = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, m2l_size*sizeof(CL_TYPE2));
    }
    CL_ERROR_CATCH

    //write the M2L coefficients to the GPU
    std::complex<double>* m2lptr = &(fRawM2LCoeff[0]);
    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fM2LCoeffBufferCL, CL_TRUE, 0, m2l_size*sizeof(CL_TYPE2), m2lptr);

    #ifdef ENFORCE_CL_FINISH
    KOpenCLInterface::GetInstance()->GetQueue().finish();
    #endif

    //swap raw m2l array with empty vector, since it is no longer used
    std::vector< std::complex<double> > temp;
    temp.swap(fRawM2LCoeff);

    //create the buffer for the normalization coefficients
    size_t norm_size = fNTerms*fNTerms;
    CL_ERROR_TRY
    {
        fNormalizationCoeffBufferCL
        = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, norm_size*sizeof(CL_TYPE2));
    }
    CL_ERROR_CATCH

    //write the buffer containing the normalization coefficients
    std::complex<double>* ptr = &(fNormalizationCoeff[0]);
    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fNormalizationCoeffBufferCL, CL_TRUE, 0, norm_size*sizeof(CL_TYPE2), ptr);

    #ifdef ENFORCE_CL_FINISH
    KOpenCLInterface::GetInstance()->GetQueue().finish();
    #endif

    //--------------------------------------

    //create the workspace buffer
    size_t workspace_size = fStride*fTotalSpatialSize;
    CL_ERROR_TRY
    {
        fWorkspaceBufferCL
        = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_WRITE, workspace_size*sizeof(CL_TYPE2));
    }
    CL_ERROR_CATCH

    //get the pointer to the DFT calculators GPU data buffer
    //we will use this to directly fill the buffer with the multipoles, and local coefficients
    //for FFTs while it is still on the GPU


    fFFTDataBufferCL = fDFTCalcOpenCL->GetDataBuffer();

    //create the reversed index look-up buffer
    CL_ERROR_TRY
    {
        fReversedIndexArrayBufferCL
        = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, fTotalSpatialSize*sizeof(unsigned int));
    }
    CL_ERROR_CATCH

    //fill the reversed index buffer
    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fReversedIndexArrayBufferCL, CL_TRUE, 0, fTotalSpatialSize*sizeof(unsigned int), &(fReversedIndexArray[0]) );

    #ifdef ENFORCE_CL_FINISH
    KOpenCLInterface::GetInstance()->GetQueue().finish();
    #endif

    //--------------------------------------

    //create the scale factor buffers
    unsigned int sf_size = (fMaxTreeDepth+1)*fStride;
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
    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fSourceScaleFactorBufferCL, CL_TRUE, 0, sf_size*sizeof(CL_TYPE), &(fSourceScaleFactorArray[0]));

    #ifdef ENFORCE_CL_FINISH
    KOpenCLInterface::GetInstance()->GetQueue().finish();
    #endif

    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fTargetScaleFactorBufferCL, CL_TRUE, 0, sf_size*sizeof(CL_TYPE), &(fTargetScaleFactorArray[0]));

    #ifdef ENFORCE_CL_FINISH
    KOpenCLInterface::GetInstance()->GetQueue().finish();
    #endif

    //---------------------------------------

    CL_ERROR_TRY
    {
        fMultipoleNodeIDListBufferCL
        = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, fMultipoleNodeIDList.size()*sizeof(unsigned int) );
    }
    CL_ERROR_CATCH

    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fMultipoleNodeIDListBufferCL, CL_TRUE, 0, fMultipoleNodeIDList.size()*sizeof(unsigned int), &(fMultipoleNodeIDList[0]));

    #ifdef ENFORCE_CL_FINISH
    KOpenCLInterface::GetInstance()->GetQueue().finish();
    #endif


    CL_ERROR_TRY
    {
        fMultipoleBlockSetIDListBufferCL
        = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, fMultipoleBlockSetIDList.size()*sizeof(unsigned int) );
    }
    CL_ERROR_CATCH

    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fMultipoleBlockSetIDListBufferCL, CL_TRUE, 0, fMultipoleBlockSetIDList.size()*sizeof(unsigned int), &(fMultipoleBlockSetIDList[0]));

    #ifdef ENFORCE_CL_FINISH
    KOpenCLInterface::GetInstance()->GetQueue().finish();
    #endif

    CL_ERROR_TRY
    {
        fPrimaryNodeIDListBufferCL
        = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, fPrimaryNodeIDList.size()*sizeof(unsigned int) );
    }
    CL_ERROR_CATCH

    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fPrimaryNodeIDListBufferCL, CL_TRUE, 0, fPrimaryNodeIDList.size()*sizeof(unsigned int), &(fPrimaryNodeIDList[0]));

    #ifdef ENFORCE_CL_FINISH
    KOpenCLInterface::GetInstance()->GetQueue().finish();
    #endif

    CL_ERROR_TRY
    {
        fPrimaryBlockSetIDListBufferCL
        = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_ONLY, fPrimaryBlockSetIDList.size()*sizeof(unsigned int) );
    }
    CL_ERROR_CATCH

    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fPrimaryBlockSetIDListBufferCL, CL_TRUE, 0, fPrimaryBlockSetIDList.size()*sizeof(unsigned int), &(fPrimaryBlockSetIDList[0]));

    #ifdef ENFORCE_CL_FINISH
    KOpenCLInterface::GetInstance()->GetQueue().finish();
    #endif
}

////////////////////////////////////////////////////////////////////////////////

void
KFMElectrostaticRemoteToLocalConverter_OpenCL::AssignBuffers()
{

//ElectrostaticRemoteToLocalCopyAndScale(const unsigned int n_moment_sets,
//                                       const unsigned int term_stride,
//                                       const unsigned int spatial_stride,
//                                       const unsigned int tree_level,
//                                       const unsigned int parent_node_start_index,
//                                       __constant const CL_TYPE* scale_factor_array,
//                                       __global unsigned int* node_ids,
//                                       __global unsigned int* block_set_ids,
//                                       __global CL_TYPE2* node_moments,
//                                       __global CL_TYPE2* block_moments)

    fCopyAndScaleKernel->setArg(0, 0); //must be reset on each node
    fCopyAndScaleKernel->setArg(1, fStride);
    fCopyAndScaleKernel->setArg(2, fTotalSpatialSize);
    fCopyAndScaleKernel->setArg(3, 0); //must be reset on each node
    fCopyAndScaleKernel->setArg(4, 0); //must be reset on each node
    fCopyAndScaleKernel->setArg(5, *fSourceScaleFactorBufferCL);
    fCopyAndScaleKernel->setArg(6, *fMultipoleNodeIDListBufferCL);
    fCopyAndScaleKernel->setArg(7, *fMultipoleBlockSetIDListBufferCL);
    fCopyAndScaleKernel->setArg(8, *fNodeRemoteMomentBufferCL);
    fCopyAndScaleKernel->setArg(9, *fFFTDataBufferCL);

//ReducedScalarMomentRemoteToLocalConverter(const unsigned int total_array_size, //total number of threads
//                                          const unsigned int degree, //expansion degree
//                                          const unsigned int spatial_stride,
//                                          __global CL_TYPE2* remote_moments,
//                                          __global CL_TYPE2* response_functions,
//                                          __global CL_TYPE2* local_moments,
//                                          __global CL_TYPE2* normalization,
//                                          __global unsigned int* reversed_index)

    fTransformationKernel->setArg(0,fStride*fTotalSpatialSize);
    fTransformationKernel->setArg(1,fDegree);
    fTransformationKernel->setArg(2,fTotalSpatialSize);
    fTransformationKernel->setArg(3,*fFFTDataBufferCL);
    fTransformationKernel->setArg(4,*fM2LCoeffBufferCL);
    fTransformationKernel->setArg(5,*fWorkspaceBufferCL);
    fTransformationKernel->setArg(6,*fNormalizationCoeffBufferCL);
    fTransformationKernel->setArg(7,*fReversedIndexArrayBufferCL);

//ElectrostaticRemoteToLocalReduceAndScale(const unsigned int n_moment_sets,
//                                         const unsigned int term_stride,
//                                         const unsigned int spatial_stride,
//                                         const unsigned int tree_level,
//                                         const unsigned int parent_node_start_index,
//                                         __constant const CL_TYPE* scale_factor_array,
//                                         __global unsigned int* node_ids,
//                                         __global unsigned int* block_set_ids,
//                                         __global CL_TYPE2* node_moments,
//                                         __global CL_TYPE2* transformed_child_moments)

    fReduceAndScaleKernel->setArg(0,0); //must be reset on each node
    fReduceAndScaleKernel->setArg(1, fStride);
    fReduceAndScaleKernel->setArg(2, fTotalSpatialSize);
    fReduceAndScaleKernel->setArg(3, 0); //must be reset on each node
    fReduceAndScaleKernel->setArg(4, 0); //must be reset on each node
    fReduceAndScaleKernel->setArg(5, *fTargetScaleFactorBufferCL);
    fReduceAndScaleKernel->setArg(6, *fPrimaryNodeIDListBufferCL);
    fReduceAndScaleKernel->setArg(7, *fPrimaryBlockSetIDListBufferCL);
    fReduceAndScaleKernel->setArg(8, *fNodeLocalMomentBufferCL);
    fReduceAndScaleKernel->setArg(9, *fFFTDataBufferCL);
}

////////////////////////////////////////////////////////////////////////////////

void
KFMElectrostaticRemoteToLocalConverter_OpenCL::ApplyCopyAndScaleKernel(unsigned int n_moment_sets)
{
    unsigned int n_global;
    unsigned int nDummy;

    //first we have to zero out the multipole moment buffer
    fZeroComplexArrayKernel->setArg(0, fStride*fTotalSpatialSize);
    fZeroComplexArrayKernel->setArg(1, *fFFTDataBufferCL);

    //compute size of the array
    n_global = fStride*fTotalSpatialSize;

    nDummy = fNZeroComplexArrayLocal - (n_global%fNZeroComplexArrayLocal);
    if(nDummy == fNZeroComplexArrayLocal){nDummy = 0;};
    n_global += nDummy;

    //now enqueue the kernel
    KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fZeroComplexArrayKernel, cl::NullRange, cl::NDRange(n_global), cl::NDRange(fNZeroComplexArrayLocal), NULL, NULL);

    #ifdef ENFORCE_CL_FINISH
    KOpenCLInterface::GetInstance()->GetQueue().finish();
    #endif

    //compute size of the array
    n_global = fStride*n_moment_sets;

    //rescale the multipoles
    //pad out n-global to be a multiple of the n-local
    nDummy = fNCopyAndScaleLocal - (n_global%fNCopyAndScaleLocal);
    if(nDummy == fNCopyAndScaleLocal){nDummy = 0;};
    n_global += nDummy;
    cl::NDRange global(n_global);
    cl::NDRange local(fNCopyAndScaleLocal);

    //now enqueue the kernel
    KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fCopyAndScaleKernel, cl::NullRange, global, local, NULL, NULL);

    #ifdef ENFORCE_CL_FINISH
    KOpenCLInterface::GetInstance()->GetQueue().finish();
    #endif

}

////////////////////////////////////////////////////////////////////////////////

void
KFMElectrostaticRemoteToLocalConverter_OpenCL::ApplyTransformationKernel()
{
    //first perform the forward dft on all the multipole coefficients
    fDFTCalcOpenCL->SetForward();
    fDFTCalcOpenCL->ExecuteOperation();

    //compute size of the array
    unsigned int array_size = fStride*fTotalSpatialSize;
    unsigned int n_global = array_size;

    //pad out n-global to be a multiple of the n-local
    unsigned int nDummy = fNTransformationLocal - (n_global%fNTransformationLocal);
    if(nDummy == fNTransformationLocal){nDummy = 0;};
    n_global += nDummy;

    cl::NDRange global(n_global);
    cl::NDRange local(fNTransformationLocal);

    KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fTransformationKernel, cl::NullRange, global, local, NULL, NULL);

    #ifdef ENFORCE_CL_FINISH
    KOpenCLInterface::GetInstance()->GetQueue().finish();
    #endif

    //now copy the workspace data (pre-x-formed local) into the FFT buffer to perform inverse DFT
    KOpenCLInterface::GetInstance()->GetQueue().enqueueCopyBuffer(*fWorkspaceBufferCL, *fFFTDataBufferCL, size_t(0), size_t(0), array_size*sizeof(CL_TYPE2) );

    #ifdef ENFORCE_CL_FINISH
    KOpenCLInterface::GetInstance()->GetQueue().finish();
    #endif

    //now perform an inverse DFT on the x-formed local
    //coefficients to get the actual local coeff
    fDFTCalcOpenCL->SetBackward();
    fDFTCalcOpenCL->ExecuteOperation();
}



////////////////////////////////////////////////////////////////////////////////

void
KFMElectrostaticRemoteToLocalConverter_OpenCL::ApplyReduceAndScaleKernel(unsigned int n_moment_sets)
{
    //compute size of the array
    unsigned int n_global = fStride*n_moment_sets;

    //pad out n-global to be a multiple of the n-local
    unsigned int nDummy = fNReduceAndScaleLocal - (n_global%fNReduceAndScaleLocal);
    if(nDummy == fNReduceAndScaleLocal){nDummy = 0;};
    n_global += nDummy;
    cl::NDRange global(n_global);
    cl::NDRange local(fNReduceAndScaleLocal);

    //now enqueue the kernel
    KOpenCLInterface::GetInstance()->GetQueue().enqueueNDRangeKernel(*fReduceAndScaleKernel, cl::NullRange, global, local, NULL, NULL);

    #ifdef ENFORCE_CL_FINISH
    KOpenCLInterface::GetInstance()->GetQueue().finish();
    #endif
}


////////////////////////////////////////////////////////////////////////////////


void
KFMElectrostaticRemoteToLocalConverter_OpenCL::CheckDeviceProperites()
{
    size_t max_buffer_size = KOpenCLInterface::GetInstance()->GetDevice().getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
    size_t total_mem_size =  KOpenCLInterface::GetInstance()->GetDevice().getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();

    //size of the response functions
    size_t m2l_size = fNResponseTerms*fTotalSpatialSize;

    if(m2l_size*sizeof(CL_TYPE2) > max_buffer_size)
    {
        //we cannot fit response_functions entirely on the gpu
        //even if we use multiple buffers
        size_t size_to_alloc_mb = ( m2l_size*sizeof(CL_TYPE2) )/(1024*1024);
        size_t max_size_mb = max_buffer_size/(1024*1024);
        size_t total_size_mb = total_mem_size/(1024*1024);

        kfmout<<"KFMElectrostaticRemoteToLocalConverter_OpenCL::CheckDeviceProperites: Error. Cannot allocate buffer of size: "<<size_to_alloc_mb<<" MB on a device with max allowable buffer size of: "<<max_size_mb<<" MB and total device memory of: "<<total_size_mb<<" MB."<<kfmendl;
        kfmexit(1);
    }
}

////////////////////////////////////////////////////////////////////////////////

void
KFMElectrostaticRemoteToLocalConverter_OpenCL::ConstructCopyAndScaleKernel()
{
    //Get name of kernel source file
    std::stringstream clFile;
    clFile << KOpenCLInterface::GetInstance()->GetKernelPath() << "/kEMField_KFMElectrostaticRemoteToLocalCopyAndScale_kernel.cl";

    KOpenCLKernelBuilder k_builder;
    fCopyAndScaleKernel = k_builder.BuildKernel(clFile.str(), std::string("ElectrostaticRemoteToLocalCopyAndScale") );

    //get n-local
    fNCopyAndScaleLocal = fCopyAndScaleKernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(KOpenCLInterface::GetInstance()->GetDevice());
}

////////////////////////////////////////////////////////////////////////////////

void
KFMElectrostaticRemoteToLocalConverter_OpenCL::ConstructTransformationKernel()
{
    //Get name of kernel source file
    std::stringstream clFile;
    clFile << KOpenCLInterface::GetInstance()->GetKernelPath() << "/kEMField_KFMReducedScalarMomentRemoteToLocalConverter_kernel.cl";

    KOpenCLKernelBuilder k_builder;
    fTransformationKernel = k_builder.BuildKernel(clFile.str(), std::string("ReducedScalarMomentRemoteToLocalConverter") );

    //get n-local
    fNTransformationLocal = fTransformationKernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(KOpenCLInterface::GetInstance()->GetDevice());
}

////////////////////////////////////////////////////////////////////////////////

void
KFMElectrostaticRemoteToLocalConverter_OpenCL::ConstructReduceAndScaleKernel()
{
    //Get name of kernel source file
    std::stringstream clFile;
    clFile << KOpenCLInterface::GetInstance()->GetKernelPath() << "/kEMField_KFMElectrostaticRemoteToLocalReduceAndScale_kernel.cl";

    KOpenCLKernelBuilder k_builder;
    fReduceAndScaleKernel = k_builder.BuildKernel(clFile.str(), std::string("ElectrostaticRemoteToLocalReduceAndScale") );

    //get n-local
    fNReduceAndScaleLocal = fReduceAndScaleKernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(KOpenCLInterface::GetInstance()->GetDevice());
}

////////////////////////////////////////////////////////////////////////////////

void
KFMElectrostaticRemoteToLocalConverter_OpenCL::ConstructZeroComplexArrayKernel()
{
    //Get name of kernel source file
    std::stringstream clFile;
    clFile << KOpenCLInterface::GetInstance()->GetKernelPath() << "/kEMField_KFMZeroComplexArray_kernel.cl";

    KOpenCLKernelBuilder k_builder;
    fZeroComplexArrayKernel = k_builder.BuildKernel(clFile.str(), std::string("ZeroComplexArray") );

    //get n-local
    fNZeroComplexArrayLocal =  fZeroComplexArrayKernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(KOpenCLInterface::GetInstance()->GetDevice());

}

////////////////////////////////////////////////////////////////////////////////

}//end of kemfield namespace
