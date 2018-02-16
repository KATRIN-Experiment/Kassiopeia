#include "KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL.hh"

#include "KFMNodeFlagValueInspector.hh"
#include "KFMEmptyIdentitySetRemover.hh"
#include "KFMLeafConditionActor.hh"
#include "KFMCompoundInspectingActor.hh"
#include "KFMLevelConditionActor.hh"

#include "KFMMatrixOperations.hh"
#include "KFMMatrixVectorOperations.hh"
#include "KFMVectorOperations.hh"

#include "KFMDenseBlockSparseMatrix.hh"
#include "KEMChunkedFileInterface.hh"
#include "KEMFileInterface.hh"
#include "KFMWorkLoadBalanceWeights.hh"

#include "KFMBatchedMultidimensionalFastFourierTransform_OpenCL.hh"

#ifdef KEMFIELD_USE_MPI
    #include "KMPIInterface.hh"
#endif

#ifdef KEMFIELD_USE_REALTIME_CLOCK
#include <time.h>
#endif

namespace KEMField
{

const std::string
KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::fWeightFilePrefix = std::string("spoclwlw_");

KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL()
{
    fTree = NULL;
    fContainer = NULL;

    fDegree = 0;
    fDivisions = 0;
    fTopLevelDivisions = 0;
    fZeroMaskSize = 0;
    fMaximumTreeDepth = 0;
    fVerbosity = 0;

    fM2MConverter = NULL;
    fM2LConverterInterface = NULL;
    fL2LConverter = NULL;
    fM2MConverter_Batched = NULL;
    fM2LConverterInterface_Batched = NULL;
    fL2LConverter_Batched = NULL;

    fUseBatchedKernels = false;
    if(KOpenCLInterface::GetInstance()->GetDevice().getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_ACCELERATOR)
    {
        fUseBatchedKernels = true;
    }

    fLocalCoeffInitializer = new KFMElectrostaticLocalCoefficientInitializer();
    fMultipoleInitializer = new KFMElectrostaticMultipoleInitializer();

    fLocalCoeffResetter = new KFMElectrostaticLocalCoefficientResetter();
    fMultipoleResetter = new KFMElectrostaticMultipoleResetter();

    fMultipoleCalculator = new KFMElectrostaticMultipoleCalculator_OpenCL();

    if(fUseBatchedKernels)
    {
        fM2MConverter_Batched = new KFMElectrostaticBatchedRemoteToRemoteConverter_OpenCL();
        fM2LConverterInterface_Batched = new KFMRemoteToLocalConverterInterface<KFMElectrostaticNodeObjects, KFMELECTROSTATICS_DIM, KFMElectrostaticBatchedRemoteToLocalConverter_OpenCL>();
        fL2LConverter_Batched = new KFMElectrostaticBatchedLocalToLocalConverter_OpenCL();
        fM2MConverter = NULL;
        fM2LConverterInterface = NULL;
        fL2LConverter = NULL;
    }
    else
    {
        fM2MConverter_Batched = NULL;
        fM2LConverterInterface_Batched = NULL;
        fL2LConverter_Batched = NULL;
        fM2MConverter = new KFMElectrostaticRemoteToRemoteConverter_OpenCL();
        fM2LConverterInterface = new KFMRemoteToLocalConverterInterface<KFMElectrostaticNodeObjects, KFMELECTROSTATICS_DIM, KFMElectrostaticRemoteToLocalConverter_OpenCL>();
        fL2LConverter = new KFMElectrostaticLocalToLocalConverter_OpenCL();
    }

};

KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::~KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL()
{
    delete fLocalCoeffInitializer;
    delete fMultipoleInitializer;
    delete fLocalCoeffResetter;
    delete fMultipoleResetter;
    delete fMultipoleCalculator;
    delete fM2MConverter;
    delete fM2LConverterInterface;
    delete fL2LConverter;
    delete fM2MConverter_Batched;
    delete fM2LConverterInterface_Batched;
    delete fL2LConverter_Batched;

    delete fMultipoleBufferCL;
    delete fLocalCoeffBufferCL;
}

void
KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::SetTree(KFMElectrostaticTree* tree)
{
    fTree = tree;
    KFMCube<KFMELECTROSTATICS_DIM>* world_cube;
    world_cube =  KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCube<KFMELECTROSTATICS_DIM> >::GetNodeObject(tree->GetRootNode());
    fWorldLength = world_cube->GetLength();

    fMultipoleCalculator->SetTree(fTree);

    if(fUseBatchedKernels)
    {
        fM2MConverter_Batched->SetTree(fTree);
        fM2LConverterInterface_Batched->SetLength(fWorldLength);
        fM2LConverterInterface_Batched->GetTopLevelM2LConverter()->SetTree(fTree);
        fM2LConverterInterface_Batched->GetTreeM2LConverter()->SetTree(fTree);
        fL2LConverter_Batched->SetTree(fTree);
    }
    else
    {
        fM2MConverter->SetTree(fTree);
        fM2LConverterInterface->SetLength(fWorldLength);
        fM2LConverterInterface->GetTopLevelM2LConverter()->SetTree(fTree);
        fM2LConverterInterface->GetTreeM2LConverter()->SetTree(fTree);
        fL2LConverter->SetTree(fTree);
    }
};

//set parameters
void
KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::SetParameters(KFMElectrostaticParameters params)
{
    fDegree = params.degree;
    fVerbosity = params.verbosity;
    fNTerms = (fDegree + 1)*(fDegree + 1);
    fDivisions = params.divisions;
    fTopLevelDivisions = params.top_level_divisions;
    fZeroMaskSize = params.zeromask;
    fMaximumTreeDepth = params.maximum_tree_depth;
    fVerbosity = params.verbosity;
    //number of non-redundant terms in a multipole/local expansion
    fNReducedTerms = ( (fDegree+1)*(fDegree+2) )/2;

    fMultipoleCalculator->SetParameters(params);

    if(fUseBatchedKernels)
    {
        fM2MConverter_Batched->SetParameters(params);
        fL2LConverter_Batched->SetParameters(params);
    }
    else
    {
        fM2MConverter->SetParameters(params);
        fL2LConverter->SetParameters(params);
    }

    if(fVerbosity > 4)
    {
        //print the parameters
        kfmout<<"KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::SetParameters: top level divisions set to "<<params.top_level_divisions<<kfmendl;
        kfmout<<"KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::SetParameters: divisions set to "<<params.divisions<<kfmendl;
        kfmout<<"KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::SetParameters: degree set to "<<params.degree<<kfmendl;
        kfmout<<"KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::SetParameters: zero mask size set to "<<params.zeromask<<kfmendl;
        kfmout<<"KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::SetParameters: max tree depth set to "<<params.maximum_tree_depth<<kfmendl;
    }
}


void
KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::Initialize()
{
    PrepareNodeSets();
    AllocateBuffers();

    fLocalCoeffInitializer->SetNumberOfTermsInSeries(fNTerms);
    fMultipoleInitializer->SetNumberOfTermsInSeries(fNTerms);

    fLocalCoeffResetter->SetNumberOfTermsInSeries(fNTerms);
    fMultipoleResetter->SetNumberOfTermsInSeries(fNTerms);

    fMultipoleCalculator->SetElectrostaticElementContainer(fContainer);
    fMultipoleCalculator->SetMultipoleNodeSet(&fNonZeroMultipoleMomentNodes);
    fMultipoleCalculator->SetNodeMomentBuffer(fMultipoleBufferCL);
    fMultipoleCalculator->Initialize();

    if(fVerbosity > 4)
    {
        kfmout<<"KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::Initialize: Initializing the multipole to multipole translator. ";
    }

    if(fUseBatchedKernels)
    {
        fM2MConverter_Batched->SetMultipoleNodeSet(&fNonZeroMultipoleMomentNodes);
        fM2MConverter_Batched->SetNodeMomentBuffer(fMultipoleBufferCL);
        fM2MConverter_Batched->Initialize();
    }
    else
    {
        fM2MConverter->SetMultipoleNodeSet(&fNonZeroMultipoleMomentNodes);
        fM2MConverter->SetNodeMomentBuffer(fMultipoleBufferCL);
        fM2MConverter->Initialize();
    }

    if(fVerbosity > 4)
    {
        kfmout<<"Done."<<kfmendl;
    }

    if(fVerbosity > 4)
    {
        kfmout<<"KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::Initialize: Initializing the multipole to local translator. ";
    }

    if(fUseBatchedKernels)
    {
        fM2LConverterInterface_Batched->SetMaxTreeDepth(fMaximumTreeDepth);
        fM2LConverterInterface_Batched->SetNumberOfTermsInSeries(fNTerms);
        fM2LConverterInterface_Batched->SetZeroMaskSize(fZeroMaskSize);
        fM2LConverterInterface_Batched->SetDivisions(fDivisions);
        fM2LConverterInterface_Batched->SetTopLevelDivisions(fTopLevelDivisions);

        fM2LConverterInterface_Batched->GetTopLevelM2LConverter()->SetMultipoleNodeSet(&fNonZeroMultipoleMomentNodes);
        fM2LConverterInterface_Batched->GetTopLevelM2LConverter()->SetPrimaryNodeSet(&fPrimaryNodes);
        fM2LConverterInterface_Batched->GetTopLevelM2LConverter()->SetNodeRemoteMomentBuffer(fMultipoleBufferCL);
        fM2LConverterInterface_Batched->GetTopLevelM2LConverter()->SetNodeLocalMomentBuffer(fLocalCoeffBufferCL);

        fM2LConverterInterface_Batched->GetTreeM2LConverter()->SetMultipoleNodeSet(&fNonZeroMultipoleMomentNodes);
        fM2LConverterInterface_Batched->GetTreeM2LConverter()->SetPrimaryNodeSet(&fPrimaryNodes);
        fM2LConverterInterface_Batched->GetTreeM2LConverter()->SetNodeRemoteMomentBuffer(fMultipoleBufferCL);
        fM2LConverterInterface_Batched->GetTreeM2LConverter()->SetNodeLocalMomentBuffer(fLocalCoeffBufferCL);

        fM2LConverterInterface_Batched->Initialize();
    }
    else
    {
        fM2LConverterInterface->SetMaxTreeDepth(fMaximumTreeDepth);
        fM2LConverterInterface->SetNumberOfTermsInSeries(fNTerms);
        fM2LConverterInterface->SetZeroMaskSize(fZeroMaskSize);
        fM2LConverterInterface->SetDivisions(fDivisions);
        fM2LConverterInterface->SetTopLevelDivisions(fTopLevelDivisions);


        fM2LConverterInterface->GetTopLevelM2LConverter()->SetMultipoleNodeSet(&fNonZeroMultipoleMomentNodes);
        fM2LConverterInterface->GetTopLevelM2LConverter()->SetPrimaryNodeSet(&fPrimaryNodes);
        fM2LConverterInterface->GetTopLevelM2LConverter()->SetNodeRemoteMomentBuffer(fMultipoleBufferCL);
        fM2LConverterInterface->GetTopLevelM2LConverter()->SetNodeLocalMomentBuffer(fLocalCoeffBufferCL);

        fM2LConverterInterface->GetTreeM2LConverter()->SetMultipoleNodeSet(&fNonZeroMultipoleMomentNodes);
        fM2LConverterInterface->GetTreeM2LConverter()->SetPrimaryNodeSet(&fPrimaryNodes);
        fM2LConverterInterface->GetTreeM2LConverter()->SetNodeRemoteMomentBuffer(fMultipoleBufferCL);
        fM2LConverterInterface->GetTreeM2LConverter()->SetNodeLocalMomentBuffer(fLocalCoeffBufferCL);

        fM2LConverterInterface->Initialize();
    }

    if(fVerbosity > 4)
    {
        kfmout<<"Done."<<kfmendl;
    }

    if(fVerbosity > 4)
    {
        kfmout<<"KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::Initialize: Initializing the local to local translator. ";
    }

    if(fUseBatchedKernels)
    {
        fL2LConverter_Batched->SetPrimaryNodeSet(&fPrimaryNodes);
        fL2LConverter_Batched->SetNodeMomentBuffer(fLocalCoeffBufferCL);
        fL2LConverter_Batched->Initialize();
    }
    else
    {
        fL2LConverter->SetPrimaryNodeSet(&fPrimaryNodes);
        fL2LConverter->SetNodeMomentBuffer(fLocalCoeffBufferCL);
        fL2LConverter->Initialize();
    }

    if(fVerbosity > 4)
    {
        kfmout<<"Done."<<kfmendl;
    }

}


void
KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::MapField()
{
    ResetMultipoleMoments();
    ResetLocalCoefficients();
    ComputeMultipoleMoments();
    ComputeLocalCoefficients();
}


void
KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::InitializeMultipoleMoments()
{
    //all node multipole coefficients reside in the GPU device buffer
    //so we do not need to allocate them on the host side

    //remove any pre-existing multipole expansions
    KFMNodeObjectRemover<KFMElectrostaticNodeObjects, KFMElectrostaticMultipoleSet> remover;
    fTree->ApplyCorecursiveAction(&remover);

    if(fVerbosity > 4)
    {
        kfmout<<"KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::InitializeMultipoleMoments: Done initializing multipole moment expansions."<<kfmendl;
    }
}



void
KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::InitializeLocalCoefficientsForPrimaryNodes()
{
    //sparse initialization only for primary nodes
    KFMElectrostaticLocalCoefficientInitializer localCoeffInitializer;
    localCoeffInitializer.SetNumberOfTermsInSeries(fNTerms);

    //delete all pre-existing local coefficient expansions
    KFMNodeObjectRemover<KFMElectrostaticNodeObjects, KFMElectrostaticLocalCoefficientSet> remover;
    fTree->ApplyCorecursiveAction(&remover);

    //the primacy flag inspector
    KFMNodeFlagValueInspector<KFMElectrostaticNodeObjects, KFMELECTROSTATICS_FLAGS> primacy_condition;
    primacy_condition.SetFlagIndex(0);
    primacy_condition.SetFlagValue(1);

    //now we constuct the conditional actor
    KFMConditionalActor< KFMElectrostaticNode > conditional_actor;
    conditional_actor.SetInspectingActor(&primacy_condition);
    conditional_actor.SetOperationalActor(&localCoeffInitializer);

    //initialize the local coefficient expansions of the primary nodes
    fTree->ApplyCorecursiveAction(&conditional_actor);

    if(fVerbosity > 4)
    {
        kfmout<<"KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::InitializeLocalCoefficientsForPrimaryNodes: Done initializing local coefficient expansions."<<kfmendl;
    }
}



void
KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::ResetMultipoleMoments()
{
    //reset all pre-existing multipole expansions
    fTree->ApplyCorecursiveAction(fMultipoleResetter);

    if(fVerbosity > 4)
    {
        kfmout<<"KFMElectrostaticBoundaryIntegratorEngine_OpenCL::ResetMultipoleMoments: Done reseting pre-exisiting multipole moments."<<kfmendl;
    }
}

void
KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::ComputeMultipoleMoments()
{
    fMultipoleCalculator->ComputeMoments();

    //now read out the multipole moments to the nodes
    if(fVerbosity > 4)
    {
        kfmout<<"KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::ComputeMultipoleMoments: Done processing and distributing boundary element moments."<<kfmendl;
    }

    if(fUseBatchedKernels)
    {
        fM2MConverter_Batched->Prepare();
        fTree->ApplyCorecursiveAction(fM2MConverter_Batched, false); //false indicates this visitation proceeds from child to parent
        fM2MConverter_Batched->Finalize();
    }
    else
    {
        fTree->ApplyRecursiveAction(fM2MConverter, false); //false indicates this visitation proceeds from child to parent
    }

    if(fVerbosity > 4)
    {
        kfmout<<"KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::ComputeMultipoleMoments: Done performing the multipole to multipole (M2M) translations."<<kfmendl;
    }

}

void
KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::ResetLocalCoefficients()
{
    //reset all existing local coefficient expansions to zero
    fTree->ApplyCorecursiveAction(fLocalCoeffResetter);

    if(fVerbosity > 4)
    {
        kfmout<<"KFMElectrostaticTreeManager::ResetLocalCoefficients: Done resetting local coefficients."<<kfmendl;
    }
}

void
KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::ComputeMultipoleToLocal()
{
    if(fUseBatchedKernels)
    {
        //compute the local coefficients due to neighbors at the same tree level
        fM2LConverterInterface_Batched->Prepare();

        //action must be applied must be corecursively in order for node batching to be effective
        fTree->ApplyCorecursiveAction(fM2LConverterInterface_Batched);

        if(fVerbosity > 4)
        {
            kfmout<<"KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::ComputeLocalCoefficients: Done performing the multipole to local (M2L) translations."<<kfmendl;
        }

        fM2LConverterInterface_Batched->Finalize();
    }
    else
    {
        //compute the local coefficients due to neighbors at the same tree level
        fM2LConverterInterface->Prepare();

        fTree->ApplyCorecursiveAction(fM2LConverterInterface);

        if(fVerbosity > 4)
        {
            kfmout<<"KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::ComputeLocalCoefficients: Done performing the multipole to local (M2L) translations."<<kfmendl;
        }
    }
}

void
KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::ComputeLocalToLocal()
{
    if(fUseBatchedKernels)
    {
        //now form the downward distributions of the local coefficients
        fL2LConverter_Batched->Prepare();
        fTree->ApplyCorecursiveAction(fL2LConverter_Batched); //action must be applied must be corecursively!
        fL2LConverter_Batched->Finalize();

        if(fVerbosity > 4)
        {
            kfmout<<"KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::ComputeLocalCoefficients: Done performing the local to local (L2L) translations."<<kfmendl;
        }
    }
    else
    {
        //now form the downward distributions of the local coefficients
        fTree->ApplyCorecursiveAction(fL2LConverter); //action must be applied must be corecursively!
        fL2LConverter->Finalize();

        if(fVerbosity > 4)
        {
            kfmout<<"KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::ComputeLocalCoefficients: Done performing the local to local (L2L) translations."<<kfmendl;
        }
    }

}


void
KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::ComputeLocalCoefficients()
{
    ComputeMultipoleToLocal();
    ComputeLocalToLocal();
}


void
KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::PrepareNodeSets()
{
    KFMCubicSpaceTreeProperties<KFMELECTROSTATICS_DIM>* tree_prop = fTree->GetTreeProperties();
    unsigned int n_nodes = tree_prop->GetNNodes();

    //first we want to determine the set of nodes with non-zero multipole moment
    fNonZeroMultipoleMomentNodes.SetTotalNumberOfNodes(n_nodes);

   //flag inspector determines if a node has multipole moments or not
    KFMNodeFlagValueInspector<KFMElectrostaticNodeObjects, KFMELECTROSTATICS_FLAGS> multipole_flag_condition;
    multipole_flag_condition.SetFlagIndex(1);
    multipole_flag_condition.SetFlagValue(1);

    KFMSpecialNodeSetCreator<KFMElectrostaticNodeObjects> multipole_set_creator;
    multipole_set_creator.SetSpecialNodeSet(&fNonZeroMultipoleMomentNodes);

    //constuct the conditional actor
    KFMConditionalActor< KFMNode<KFMElectrostaticNodeObjects> > multipole_conditional_actor;
    multipole_conditional_actor.SetInspectingActor(&multipole_flag_condition);
    multipole_conditional_actor.SetOperationalActor(&multipole_set_creator);

    fTree->ApplyCorecursiveAction(&multipole_conditional_actor);

////////////////////////////////////////////////////////////////////////////////

    //now we want to determine the set of primary nodes
    fPrimaryNodes.SetTotalNumberOfNodes(n_nodes);

    //flag inspector determines if a node is primary or not
    KFMNodeFlagValueInspector<KFMElectrostaticNodeObjects, KFMELECTROSTATICS_FLAGS> primary_flag_condition;
    primary_flag_condition.SetFlagIndex(0);
    primary_flag_condition.SetFlagValue(1);

    KFMSpecialNodeSetCreator<KFMElectrostaticNodeObjects> primary_set_creator;
    primary_set_creator.SetSpecialNodeSet(&fPrimaryNodes);

    //now we constuct the conditional actor
    KFMConditionalActor< KFMNode<KFMElectrostaticNodeObjects> > primary_conditional_actor;
    primary_conditional_actor.SetInspectingActor(&primary_flag_condition);
    primary_conditional_actor.SetOperationalActor(&primary_set_creator);

    fTree->ApplyCorecursiveAction(&primary_conditional_actor);

    //now get the number of non-zero multipole, and primary nodes

    fNMultipoleNodes = fNonZeroMultipoleMomentNodes.GetSize();
    fNPrimaryNodes = fPrimaryNodes.GetSize();

////////////////////////////////////////////////////////////////////////////////

    //now determine the primary nodes in the top level
    fTopLevelPrimaryNodes.SetTotalNumberOfNodes(fTree->GetRootNode()->GetNChildren()+1);

    //the node level condition
    KFMLevelConditionActor< KFMElectrostaticNode > level_condition;
    level_condition.SetLevel(1);
    level_condition.SetEqualOrLessThan();

    //compound inspecting actor
    KFMCompoundInspectingActor<KFMElectrostaticNode> compound_inspector;
    compound_inspector.UseAndCondition();
    compound_inspector.AddInspectingActor(&level_condition);
    compound_inspector.AddInspectingActor(&primary_flag_condition);

    KFMSpecialNodeSetCreator<KFMElectrostaticNodeObjects> top_level_primary_set_creator;
    top_level_primary_set_creator.SetSpecialNodeSet(&fTopLevelPrimaryNodes);

    //now we constuct the conditional actor
    KFMConditionalActor< KFMNode<KFMElectrostaticNodeObjects> > top_level_primary_conditional_actor;
    top_level_primary_conditional_actor.SetInspectingActor(&compound_inspector);
    top_level_primary_conditional_actor.SetOperationalActor(&top_level_primary_set_creator);

    fTree->ApplyCorecursiveAction(&top_level_primary_conditional_actor);

    //now get the number of non-zero multipole, and primary nodes
    fNTopLevelPrimaryNodes = fTopLevelPrimaryNodes.GetSize();
}


void
KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::RecieveTopLevelLocalCoefficients()
{
    //read the primary node local coefficients back from the gpu;
    unsigned int stride = (fDegree+1)*(fDegree+2)/2;
    unsigned int primary_size = fNTopLevelPrimaryNodes*stride;

    std::vector< std::complex<double> > primary_local_coeff;
    primary_local_coeff.resize(primary_size, std::complex<double>(0.,0.) );

    KOpenCLInterface::GetInstance()->GetQueue().enqueueReadBuffer(*fLocalCoeffBufferCL, CL_TRUE, 0, primary_size*sizeof(CL_TYPE2), &(primary_local_coeff[0]) );

    //now distribute the primary node moments
    for(unsigned int i=0; i<fNTopLevelPrimaryNodes; i++)
    {
        KFMNode<KFMElectrostaticNodeObjects>* node = fTopLevelPrimaryNodes.GetNodeFromSpecializedID(i);
        KFMElectrostaticLocalCoefficientSet* set = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMElectrostaticLocalCoefficientSet>::GetNodeObject(node);

        if(set != NULL)
        {
            std::complex<double> temp;
            //we use raw ptr for speed
            double* rmoments = &( (*(set->GetRealMoments()))[0] );
            double* imoments = &( (*(set->GetImaginaryMoments()))[0] );
            for(unsigned int j=0; j < stride; ++j)
            {
                temp = primary_local_coeff[i*stride + j];
                rmoments[j] = temp.real();
                imoments[j] = temp.imag();
            }
        }
    }

}

void
KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::SendTopLevelLocalCoefficients()
{
    //read the primary node local coefficients back from the gpu;
    unsigned int stride = (fDegree+1)*(fDegree+2)/2;
    unsigned int primary_size = fNTopLevelPrimaryNodes*stride;

    std::vector< std::complex<double> > primary_local_coeff;
    std::complex<double> zero(0.,0.);
    primary_local_coeff.resize(primary_size, zero);

    //now distribute the primary node moments
    for(unsigned int i=0; i<fNTopLevelPrimaryNodes; i++)
    {
        KFMNode<KFMElectrostaticNodeObjects>* node = fTopLevelPrimaryNodes.GetNodeFromSpecializedID(i);
        KFMElectrostaticLocalCoefficientSet* set = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMElectrostaticLocalCoefficientSet>::GetNodeObject(node);

        if(set != NULL)
        {
            std::complex<double> temp;
            //we use raw ptr for speed
            double* rmoments = &( (*(set->GetRealMoments()))[0] );
            double* imoments = &( (*(set->GetImaginaryMoments()))[0] );
            for(unsigned int j=0; j < stride; ++j)
            {
                temp.real(rmoments[j]);
                temp.imag(imoments[j]);
                primary_local_coeff[i*stride + j] = temp;
            }
        }
    }

    KOpenCLInterface::GetInstance()->GetQueue().enqueueWriteBuffer(*fLocalCoeffBufferCL, CL_TRUE, 0, primary_size*sizeof(CL_TYPE2), &(primary_local_coeff[0]) );

}


void
KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::AllocateBuffers()
{
    CheckDeviceProperites();

    //create the node multipole moment buffer
    CL_ERROR_TRY
    {
        fMultipoleBufferCL
        = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_WRITE, fNMultipoleNodes*fNReducedTerms*sizeof(CL_TYPE2));
    }
    CL_ERROR_CATCH

    //create the node local coefficient moment buffer
    CL_ERROR_TRY
    {
        fLocalCoeffBufferCL
        = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_WRITE, fNPrimaryNodes*fNReducedTerms*sizeof(CL_TYPE2));
    }
    CL_ERROR_CATCH
}

void
KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::CheckDeviceProperites()
{
    size_t max_buffer_size = KOpenCLInterface::GetInstance()->GetDevice().getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
    size_t total_mem_size =  KOpenCLInterface::GetInstance()->GetDevice().getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();

    //size of moment buffers
    size_t multipole_size = fNMultipoleNodes*fNReducedTerms*sizeof(CL_TYPE2);
    size_t local_coeff_size = fNPrimaryNodes*fNReducedTerms*sizeof(CL_TYPE2);

    if(multipole_size > max_buffer_size)
    {
        //we cannot fit multipole/local buffers on GPU
        size_t size_to_alloc_mb = ( multipole_size )/(1024*1024);
        size_t max_size_mb = max_buffer_size/(1024*1024);
        size_t total_size_mb = total_mem_size/(1024*1024);

        kfmout<<"KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::BuildBuffers: Error. Cannot allocate multipole buffer of size: "<<size_to_alloc_mb<<" MB on a device with max allowable buffer size of: "<<max_size_mb<<" MB and total device memory of: "<<total_size_mb<<" MB."<<kfmendl;
        kfmexit(1);
    }

    if(local_coeff_size > max_buffer_size)
    {
        //we cannot fit multipole/local buffers on GPU
        size_t size_to_alloc_mb = ( local_coeff_size )/(1024*1024);
        size_t max_size_mb = max_buffer_size/(1024*1024);
        size_t total_size_mb = total_mem_size/(1024*1024);

        kfmout<<"KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::BuildBuffers: Error. Cannot allocate local coefficient buffer of size: "<<size_to_alloc_mb<<" MB on a device with max allowable buffer size of: "<<max_size_mb<<" MB and total device memory of: "<<total_size_mb<<" MB."<<kfmendl;
        kfmexit(1);
    }

}


void
KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::EvaluateWorkLoads(unsigned int divisions, unsigned int zeromask)
{
    //check if we have computed these parameters before
    //we need save and re-use the weight factors because if we don't
    //the tree structure construction will become non-deterministic
    //(since the weights might change slightly depending on system load)
    //if the tree structure isn't the same then any saved matrix files we
    //may have previously computed are invalid

    std::stringstream ss;
    ss << "d" << divisions;
    ss << "z" << zeromask;

    std::string filename = fWeightFilePrefix + ss.str() + std::string(".ksa");
    bool on_file = KEMFileInterface::GetInstance()->DoesFileExist(filename);

    if(on_file)
    {
        //read the weight file from disk
        bool result = false;
        KSAObjectInputNode< KFMWorkLoadBalanceWeights >* in_node;
        in_node = new KSAObjectInputNode< KFMWorkLoadBalanceWeights >( KSAClassName< KFMWorkLoadBalanceWeights >::name() );
        KEMFileInterface::GetInstance()->ReadKSAFileFromActiveDirectory(in_node, filename, result);
        if(!result)
        {
            on_file = false;
        }
        else
        {
            fDiskWeight = in_node->GetObject()->GetDiskMatrixVectorProductWeight();
            fRamWeight = in_node->GetObject()->GetRamMatrixVectorProductWeight();
            fFFTWeight = in_node->GetObject()->GetFFTWeight();
        }
        delete in_node;
    }

    if(!on_file)
    {
        fDiskWeight = ComputeDiskMatrixVectorProductWeight();
        fRamWeight = ComputeRamMatrixVectorProductWeight();
        fFFTWeight = ComputeFFTWeight(divisions, zeromask);

        KFMWorkLoadBalanceWeights weights;
        weights.SetDivisions(divisions);
        weights.SetZeroMaskSize(zeromask);
        weights.SetDiskMatrixVectorProductWeight(fDiskWeight);
        weights.SetRamMatrixVectorProductWeight(fRamWeight);
        weights.SetFFTWeight(fFFTWeight);

        //now write out the values to disk so we can reuse them next time if needed
        KSAObjectOutputNode< KFMWorkLoadBalanceWeights >* out_node = new KSAObjectOutputNode<KFMWorkLoadBalanceWeights>( KSAClassName<KFMWorkLoadBalanceWeights>::name() );
        out_node->AttachObjectToNode(&weights);
        bool result;
        KEMFileInterface::GetInstance()->SaveKSAFileToActiveDirectory(out_node, filename, result);
        delete out_node;
    }
}


double
KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::ComputeDiskMatrixVectorProductWeight()
{
    //evaluate the time for a matrix-like chuck of multiply-adds
    //read off of a disk

    unsigned int n_samples = 100;
    unsigned int dim = KFMDenseBlockSparseMatrix<double>::GetSuggestedMaximumRowWidth();

    //create a temporary file
    std::string filename("disk_matrix_temp.wle");
    KEMChunkedFileInterface* fElementFileInterface = new KEMChunkedFileInterface();
    fElementFileInterface->OpenFileForWriting(filename);

    std::vector<double> tmp;
    size_t sz = n_samples*dim*dim;
    tmp.resize(sz, 1);
    fElementFileInterface->Write( sz, &(tmp[0]) );
    fElementFileInterface->CloseFile();

    fElementFileInterface->OpenFileForReading(filename);

    kfm_vector* lhs = kfm_vector_alloc(dim);
    kfm_vector* rhs = kfm_vector_alloc(dim);
    kfm_matrix* mx = kfm_matrix_alloc(dim, dim);

    double time;
    #ifdef KEMFIELD_USE_REALTIME_CLOCK
        timespec start, end;
        clock_gettime(CLOCK_REALTIME, &start);
    #else
        clock_t cstart, cend;
        cstart = clock();
    #endif

    for(unsigned int i=0; i<n_samples; i++)
    {
        fElementFileInterface->Read(dim*dim, &(tmp[0]) );
        kfm_matrix_vector_product(mx, rhs, lhs);
    }

    #ifdef KEMFIELD_USE_REALTIME_CLOCK
        clock_gettime(CLOCK_REALTIME, &end);
        timespec temp = diff(start, end);
        time = (double)temp.tv_sec + (double)(temp.tv_nsec*1e-9);
    #else
        cend = clock();
        time = ((double)(cend - cstart))/CLOCKS_PER_SEC; // time in seconds
    #endif

    //scale by number of samples and number of matrix elements
    double weight = time/(double)n_samples;
    weight /= ( (double)(dim*dim) );

    kfm_matrix_free(mx);
    kfm_vector_free(rhs);
    kfm_vector_free(lhs);

    fElementFileInterface->CloseFile();
    KEMFileInterface::GetInstance()->RemoveFileFromActiveDirectory(filename);

    return weight;
}

double
KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::ComputeRamMatrixVectorProductWeight()
{
    //now we evaluate the time for a matrix like chuck of multiply-adds
    unsigned int n_samples = 100;
    unsigned int dim = KFMDenseBlockSparseMatrix<double>::GetSuggestedMaximumRowWidth();

    kfm_vector* lhs = kfm_vector_alloc(dim);
    kfm_vector* rhs = kfm_vector_alloc(dim);
    kfm_matrix* mx = kfm_matrix_alloc(dim, dim);

    double time;
    #ifdef KEMFIELD_USE_REALTIME_CLOCK
        timespec start, end;
        clock_gettime(CLOCK_REALTIME, &start);
    #else
        clock_t cstart, cend;
        cstart = clock();
    #endif

    for(unsigned int i=0; i<n_samples; i++)
    {
        kfm_matrix_vector_product(mx, rhs, lhs);
    }

    #ifdef KEMFIELD_USE_REALTIME_CLOCK
        clock_gettime(CLOCK_REALTIME, &end);
        timespec temp = diff(start, end);
        time = (double)temp.tv_sec + (double)(temp.tv_nsec*1e-9);
    #else
        cend = clock();
        time = ((double)(cend - cstart))/CLOCKS_PER_SEC; // time in seconds
    #endif

    //scale by number of samples and number of matrix elements
    double weight = time/(double)n_samples;
    weight /= ( (double)(dim*dim) );

    kfm_matrix_free(mx);
    kfm_vector_free(rhs);
    kfm_vector_free(lhs);

    return weight;
}

double
KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::ComputeFFTWeight(unsigned int divisions, unsigned int zeromask)
{
    unsigned int n_samples = 100;
    unsigned int fft_batch_size = 100;
    unsigned int spatial = 2*divisions*(zeromask + 1);
    unsigned int dim_size[4] = {fft_batch_size,spatial,spatial,spatial};

    //evaluate time taken for opencl fft's
    unsigned int total_size_gpu = dim_size[0]*dim_size[1]*dim_size[2]*dim_size[3];
    std::complex<double>* raw_data_gpu = new std::complex<double>[total_size_gpu];
    KFMArrayWrapper< std::complex<double>, 4> input_gpu(raw_data_gpu, dim_size);

    KFMBatchedMultidimensionalFastFourierTransform_OpenCL<3>* fft_gpu = new KFMBatchedMultidimensionalFastFourierTransform_OpenCL<3>();

    fft_gpu->SetForward();
    fft_gpu->SetInput(&input_gpu);
    fft_gpu->SetOutput(&input_gpu);
    fft_gpu->Initialize();

    double time;
    #ifdef KEMFIELD_USE_REALTIME_CLOCK
        timespec start, end;
        clock_gettime(CLOCK_REALTIME, &start);
    #else
        clock_t cstart, cend;
        cstart = clock();
    #endif

    for(unsigned int s=0; s<n_samples; s++)
    {
        fft_gpu->ExecuteOperation();
    }

    #ifdef KEMFIELD_USE_REALTIME_CLOCK
        clock_gettime(CLOCK_REALTIME, &end);
        timespec temp = diff(start, end);
        time = (double)temp.tv_sec + (double)(temp.tv_nsec*1e-9);
    #else
        cend = clock();
        time = ((double)(cend - cstart))/CLOCKS_PER_SEC; // time in seconds
    #endif

    double weight = time/( (double)(n_samples*fft_batch_size) );

    delete fft_gpu;
    delete[] raw_data_gpu;

    return weight;
}


}
