#include "KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL.hh"

#include "KFMNodeFlagValueInspector.hh"
#include "KFMEmptyIdentitySetRemover.hh"
#include "KFMLeafConditionActor.hh"

#include "KFMScalarMomentRemoteToLocalConverter_OpenCL.hh"
#include "KFMReducedScalarMomentRemoteToLocalConverter_OpenCL.hh"
#include "KFMSparseReducedScalarMomentRemoteToLocalConverter_OpenCL.hh"
#include "KFMScalarMomentLocalToLocalConverter_OpenCL.hh"
#include "KFMScalarMomentRemoteToLocalConverter.hh"

namespace KEMField
{

#ifdef USE_REDUCED_M2L
typedef KFMSparseReducedScalarMomentRemoteToLocalConverter_OpenCL<KFMElectrostaticNodeObjects, KFMElectrostaticMultipoleSet, KFMElectrostaticLocalCoefficientSet, KFMResponseKernel_3DLaplaceM2L, KFMELECTROSTATICS_DIM, KFMELECTROSTATICS_FLAGS>
KFMElectrostaticRemoteToLocalConverter_OpenCL;

#else

typedef KFMScalarMomentRemoteToLocalConverter_OpenCL<KFMElectrostaticNodeObjects, KFMElectrostaticMultipoleSet, KFMElectrostaticLocalCoefficientSet, KFMResponseKernel_3DLaplaceM2L, KFMELECTROSTATICS_DIM>
KFMElectrostaticRemoteToLocalConverter_OpenCL;
#endif

typedef KFMScalarMomentLocalToLocalConverter_OpenCL<KFMElectrostaticNodeObjects, KFMElectrostaticLocalCoefficientSet, KFMResponseKernel_3DLaplaceL2L, KFMELECTROSTATICS_DIM>
KFMElectrostaticLocalToLocalConverter_OpenCL;

//typedef KFMScalarMomentRemoteToRemoteConverter_OpenCL<KFMElectrostaticNodeObjects, KFMElectrostaticMultipoleSet, KFMResponseKernel_3DLaplaceM2M, KFMELECTROSTATICS_DIM>
//KFMElectrostaticRemoteToRemoteConverter_OpenCL;


KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL()
{
    fTree = NULL;
    fContainer = NULL;

    fDegree = 0;
    fDivisions = 0;
    fZeroMaskSize = 0;
    fMaximumTreeDepth = 0;
    fVerbosity = 0;

    fLocalCoeffInitializer = new KFMElectrostaticLocalCoefficientInitializer();
    fMultipoleInitializer = new KFMElectrostaticMultipoleInitializer();

    fLocalCoeffResetter = new KFMElectrostaticLocalCoefficientResetter();
    fMultipoleResetter = new KFMElectrostaticMultipoleResetter();

    fMultipoleCalculator = new KFMElectrostaticMultipoleCalculator_OpenCL();
    fMultipoleDistributor = new KFMElectrostaticMultipoleDistributor_OpenCL();

    fM2MConverter = new KFMElectrostaticRemoteToRemoteConverter_OpenCL();
    fM2LConverter = new KFMElectrostaticRemoteToLocalConverter_OpenCL();
    fL2LConverter = new KFMElectrostaticLocalToLocalConverter_OpenCL();
};

KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::~KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL()
{
    delete fLocalCoeffInitializer;
    delete fMultipoleInitializer;
    delete fLocalCoeffResetter;
    delete fMultipoleResetter;
    delete fMultipoleCalculator;
    delete fMultipoleDistributor;
    delete fM2MConverter;
    delete fM2LConverter;
    delete fL2LConverter;
}

void
KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::SetTree(KFMElectrostaticTree* tree)
{
    fTree = tree;
    SetParameters(tree->GetParameters());
    KFMCube<KFMELECTROSTATICS_DIM>* world_cube;
    world_cube =  KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCube<KFMELECTROSTATICS_DIM> >::GetNodeObject(tree->GetRootNode());
    fWorldLength = world_cube->GetLength();

    fMultipoleCalculator->SetTree(fTree);
    fM2MConverter->SetTree(fTree);
};

//set parameters
void
KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::SetParameters(KFMElectrostaticParameters params)
{
    fDegree = params.degree;
    fVerbosity = params.verbosity;
    fNTerms = (fDegree + 1)*(fDegree + 1);
    fDivisions = params.divisions;
    fZeroMaskSize = params.zeromask;
    fMaximumTreeDepth = params.maximum_tree_depth;
    fVerbosity = params.verbosity;
    //number of non-redundant terms in a multipole/local expansion
    fNReducedTerms = (fDegree+1)*(fDegree+2)/2;


    fMultipoleDistributor->SetDegree(fDegree);

    if(fVerbosity > 2)
    {
        //print the parameters
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

    fMultipoleDistributor->SetMultipoleNodeSet(&fNonZeroMultipoleMomentNodes);
    fMultipoleDistributor->SetNodeMomentBuffer(fMultipoleBufferCL);
    fMultipoleDistributor->Initialize();

    if(fVerbosity > 2)
    {
        kfmout<<"KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::Initialize: Initializing the multipole to multipole translator. ";
    }

    fM2MConverter->SetMultipoleNodeSet(&fNonZeroMultipoleMomentNodes);
    fM2MConverter->SetNodeMomentBuffer(fMultipoleBufferCL);
    fM2MConverter->Initialize();

//    fM2MConverter->SetNumberOfTermsInSeries(fNTerms);
//    fM2MConverter->SetDivisions(fDivisions);
//    fM2MConverter->Initialize();

    if(fVerbosity > 2)
    {
        kfmout<<"Done."<<kfmendl;
    }

    if(fVerbosity > 2)
    {
        kfmout<<"KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::Initialize: Initializing the multipole to local translator. ";
    }

    fM2LConverter->SetLength(fWorldLength);
    fM2LConverter->SetMaxTreeDepth(fMaximumTreeDepth);
    fM2LConverter->SetNumberOfTermsInSeries(fNTerms);
    fM2LConverter->SetZeroMaskSize(fZeroMaskSize);
    fM2LConverter->SetDivisions(fDivisions);
    fM2LConverter->Initialize();

    if(fVerbosity > 2)
    {
        kfmout<<"Done."<<kfmendl;
    }

    if(fVerbosity > 2)
    {
        kfmout<<"KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::Initialize: Initializing the local to local translator. ";
    }

    fL2LConverter->SetNumberOfTermsInSeries(fNTerms);
    fL2LConverter->SetDivisions(fDivisions);
    fL2LConverter->Initialize();

    if(fVerbosity > 2)
    {
        kfmout<<"Done."<<kfmendl;
    }

    InitializeMultipoleMoments();
    InitializeLocalCoefficients();
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
    //remove any pre-existing multipole expansions
    KFMNodeObjectRemover<KFMElectrostaticNodeObjects, KFMElectrostaticMultipoleSet> remover;
    fTree->ApplyCorecursiveAction(&remover);

    //condition for a node to have a multipole expansion is based on the non-zero multipole moment flag
    KFMNodeFlagValueInspector<KFMElectrostaticNodeObjects, KFMELECTROSTATICS_FLAGS> multipole_flag_condition;
    multipole_flag_condition.SetFlagIndex(1);
    multipole_flag_condition.SetFlagValue(1);

    //now we constuct the conditional actor
    KFMConditionalActor< KFMElectrostaticNode > conditional_actor;

    conditional_actor.SetInspectingActor(&multipole_flag_condition);
    conditional_actor.SetOperationalActor(fMultipoleInitializer);

    //initialize multipole expansions for appropriate nodes
    fTree->ApplyRecursiveAction(&conditional_actor);
}

void
KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::ResetMultipoleMoments()
{
    //reset all pre-existing multipole expansions
    fTree->ApplyCorecursiveAction(fMultipoleResetter);

    if(fVerbosity > 3)
    {
        kfmout<<"KFMElectrostaticTreeManager::ResetMultipoleMoments: Done reseting pre-exisiting multipole moments."<<kfmendl;
    }

}

void
KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::ComputeMultipoleMoments()
{
    fMultipoleCalculator->ComputeMoments();

//    //now read out the multipole moments to the nodes
    if(fVerbosity > 3)
    {
        kfmout<<"KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::ComputeMultipoleMoments: Done processing and distributing boundary element moments."<<kfmendl;
    }


    //now we perform the upward pass to collect child nodes' moments into their parents' moments
    fTree->ApplyRecursiveAction(fM2MConverter, false); //false indicates this visitation proceeds from child to parent

    if(fVerbosity > 3)
    {
        kfmout<<"KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::ComputeMultipoleMoments: Done performing the multipole to multipole (M2M) translations."<<kfmendl;
    }

    fMultipoleDistributor->DistributeMoments();
}


void
KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::InitializeLocalCoefficients()
{
    //sparse initialization only for primary nodes

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
    conditional_actor.SetOperationalActor(fLocalCoeffInitializer);

    //initialize the local coefficient expansions of the primary nodes
    fTree->ApplyCorecursiveAction(&conditional_actor);

    if(fVerbosity > 3)
    {
        kfmout<<"KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::InitializeLocalCoefficients: Done initializing local coefficient expansions."<<kfmendl;
    }
}

void
KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::ResetLocalCoefficients()
{
    //reset all existing local coefficient expansions to zero
    fTree->ApplyCorecursiveAction(fLocalCoeffResetter);

    if(fVerbosity > 3)
    {
        kfmout<<"KFMElectrostaticTreeManager::ResetLocalCoefficients: Done resetting local coefficients."<<kfmendl;
    }
}

void
KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::ComputeLocalCoefficients()
{
    //compute the local coefficients due to neighbors at the same tree level
    fM2LConverter->Prepare(fTree);
    do
    {
        fTree->ApplyCorecursiveAction(fM2LConverter);
    }
    while( !(fM2LConverter->IsFinished()) );
    fM2LConverter->Finalize(fTree);


    if(fVerbosity > 3)
    {
        kfmout<<"KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::ComputeLocalCoefficients: Done performing the multipole to local (M2L) translations."<<kfmendl;
    }

    //now form the downward distributions of the local coefficients
    fTree->ApplyRecursiveAction(fL2LConverter);

    if(fVerbosity > 3)
    {
        kfmout<<"KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::ComputeLocalCoefficients: Done performing the local to local (L2L) translations."<<kfmendl;
    }
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
}

void
KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::AllocateBuffers()
{
    CheckDeviceProperites();

    //create the node multipole moment buffer
    fMultipoleBufferCL
    = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_WRITE, fNMultipoleNodes*fNReducedTerms*sizeof(CL_TYPE2));

    //create the node local coefficient moment buffer
    fLocalCoeffBufferCL
    = new cl::Buffer(KOpenCLInterface::GetInstance()->GetContext(), CL_MEM_READ_WRITE, fNPrimaryNodes*fNReducedTerms*sizeof(CL_TYPE2));
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

        std::cout<<"multipoles"<<std::endl;
        kfmout<<"KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::BuildBuffers: Error. Cannot allocate multipole buffer of size: "<<size_to_alloc_mb<<" MB on a device with max allowable buffer size of: "<<max_size_mb<<" MB and total device memory of: "<<total_size_mb<<" MB."<<kfmendl;
        kfmexit(1);
    }

    if(local_coeff_size > max_buffer_size)
    {
        //we cannot fit multipole/local buffers on GPU
        size_t size_to_alloc_mb = ( local_coeff_size )/(1024*1024);
        size_t max_size_mb = max_buffer_size/(1024*1024);
        size_t total_size_mb = total_mem_size/(1024*1024);

        std::cout<<"local coeff"<<std::endl;
        kfmout<<"KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL::BuildBuffers: Error. Cannot allocate local coefficient buffer of size: "<<size_to_alloc_mb<<" MB on a device with max allowable buffer size of: "<<max_size_mb<<" MB and total device memory of: "<<total_size_mb<<" MB."<<kfmendl;
        kfmexit(1);
    }

}



}
