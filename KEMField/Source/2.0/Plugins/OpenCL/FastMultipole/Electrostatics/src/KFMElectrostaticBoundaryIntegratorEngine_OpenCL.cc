#include "KFMElectrostaticBoundaryIntegratorEngine_OpenCL.hh"

#include "KFMNodeFlagValueInspector.hh"
#include "KFMEmptyIdentitySetRemover.hh"
#include "KFMLeafConditionActor.hh"

#include "KFMElectrostaticMultipoleBatchCalculator.hh"
#include "KFMElectrostaticMultipoleBatchCalculator_OpenCL.hh"

#include "KFMScalarMomentRemoteToRemoteConverter_OpenCL.hh"
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

//typedef KFMReducedScalarMomentRemoteToLocalConverter_OpenCL<KFMElectrostaticNodeObjects, KFMElectrostaticMultipoleSet, KFMElectrostaticLocalCoefficientSet, KFMResponseKernel_3DLaplaceM2L, 3>
//KFMElectrostaticRemoteToLocalConverter_OpenCL;

#else

typedef KFMScalarMomentRemoteToLocalConverter_OpenCL<KFMElectrostaticNodeObjects, KFMElectrostaticMultipoleSet, KFMElectrostaticLocalCoefficientSet, KFMResponseKernel_3DLaplaceM2L, KFMELECTROSTATICS_DIM>
KFMElectrostaticRemoteToLocalConverter_OpenCL;
#endif

typedef KFMScalarMomentLocalToLocalConverter_OpenCL<KFMElectrostaticNodeObjects, KFMElectrostaticLocalCoefficientSet, KFMResponseKernel_3DLaplaceL2L, KFMELECTROSTATICS_DIM>
KFMElectrostaticLocalToLocalConverter_OpenCL;

typedef KFMScalarMomentRemoteToRemoteConverter_OpenCL<KFMElectrostaticNodeObjects, KFMElectrostaticMultipoleSet, KFMResponseKernel_3DLaplaceM2M, KFMELECTROSTATICS_DIM>
KFMElectrostaticRemoteToRemoteConverter_OpenCL;



KFMElectrostaticBoundaryIntegratorEngine_OpenCL::KFMElectrostaticBoundaryIntegratorEngine_OpenCL()
{
    fTree = NULL;
    fContainer = NULL;

    fDegree = 0;
    fDivisions = 0;
    fZeroMaskSize = 0;
    fMaximumTreeDepth = 0;
    fVerbosity = 0;

    #ifdef KEMFIELD_OPENCL_ANALYTIC_MULTIPOLE
    fBatchCalc = new KFMElectrostaticMultipoleBatchCalculator_OpenCL();
    #else
    fBatchCalc = new KFMElectrostaticMultipoleBatchCalculator();
    #endif

    fMultipoleDistributor = new KFMElectrostaticElementMultipoleDistributor();
    fMultipoleDistributor->SetBatchCalculator(fBatchCalc);
    fElementNodeAssociator = new KFMElectrostaticElementNodeAssociator();

    fLocalCoeffInitializer = new KFMElectrostaticLocalCoefficientInitializer();
    fMultipoleInitializer = new KFMElectrostaticMultipoleInitializer();

    fLocalCoeffResetter = new KFMElectrostaticLocalCoefficientResetter();
    fMultipoleResetter = new KFMElectrostaticMultipoleResetter();

    fM2MConverter = new KFMElectrostaticRemoteToRemoteConverter_OpenCL();
    fM2LConverter = new KFMElectrostaticRemoteToLocalConverter_OpenCL();
    fL2LConverter = new KFMElectrostaticLocalToLocalConverter_OpenCL();
};

KFMElectrostaticBoundaryIntegratorEngine_OpenCL::~KFMElectrostaticBoundaryIntegratorEngine_OpenCL()
{
    delete fBatchCalc;
    delete fMultipoleDistributor;
    delete fElementNodeAssociator;
    delete fLocalCoeffInitializer;
    delete fMultipoleInitializer;
    delete fLocalCoeffResetter;
    delete fMultipoleResetter;
    delete fM2MConverter;
    delete fM2LConverter;
    delete fL2LConverter;
}

void
KFMElectrostaticBoundaryIntegratorEngine_OpenCL::SetTree(KFMElectrostaticTree* tree)
{
    fTree = tree;
    SetParameters(tree->GetParameters());
    KFMCube<KFMELECTROSTATICS_DIM>* world_cube;
    world_cube =  KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCube<KFMELECTROSTATICS_DIM> >::GetNodeObject(tree->GetRootNode());
    fWorldLength = world_cube->GetLength();
};

//set parameters
void
KFMElectrostaticBoundaryIntegratorEngine_OpenCL::SetParameters(KFMElectrostaticParameters params)
{
    fDegree = params.degree;
    fNTerms = (fDegree + 1)*(fDegree + 1);
    fDivisions = params.divisions;
    fZeroMaskSize = params.zeromask;
    fMaximumTreeDepth = params.maximum_tree_depth;
    fVerbosity = params.verbosity;

    if(fVerbosity > 2)
    {
        //print the parameters
        kfmout<<"KFMElectrostaticBoundaryIntegratorEngine_OpenCL::SetParameters: divisions set to "<<params.divisions<<kfmendl;
        kfmout<<"KFMElectrostaticBoundaryIntegratorEngine_OpenCL::SetParameters: degree set to "<<params.degree<<kfmendl;
        kfmout<<"KFMElectrostaticBoundaryIntegratorEngine_OpenCL::SetParameters: zero mask size set to "<<params.zeromask<<kfmendl;
        kfmout<<"KFMElectrostaticBoundaryIntegratorEngine_OpenCL::SetParameters: max tree depth set to "<<params.maximum_tree_depth<<kfmendl;
    }
}


void
KFMElectrostaticBoundaryIntegratorEngine_OpenCL::Initialize()
{

    if(fVerbosity > 2)
    {
        kfmout<<"KFMElectrostaticBoundaryIntegratorEngine_OpenCL::Initialize: Initializing the element multipole moment batch calculator. ";
    }

    fBatchCalc->SetDegree(fDegree);
    fBatchCalc->SetElectrostaticElementContainer(fContainer);
    fBatchCalc->Initialize();

    if(fVerbosity > 2)
    {
        kfmout<<"Done."<<kfmendl;
    }

    fLocalCoeffInitializer->SetNumberOfTermsInSeries(fNTerms);
    fMultipoleInitializer->SetNumberOfTermsInSeries(fNTerms);

    fLocalCoeffResetter->SetNumberOfTermsInSeries(fNTerms);
    fMultipoleResetter->SetNumberOfTermsInSeries(fNTerms);

    if(fVerbosity > 2)
    {
        kfmout<<"KFMElectrostaticBoundaryIntegratorEngine_OpenCL::Initialize: Initializing the multipole to multipole translator. ";
    }

    fM2MConverter->SetNumberOfTermsInSeries(fNTerms);
    fM2MConverter->SetDivisions(fDivisions);
    fM2MConverter->Initialize();

    if(fVerbosity > 2)
    {
        kfmout<<"Done."<<kfmendl;
    }

    if(fVerbosity > 2)
    {
        kfmout<<"KFMElectrostaticBoundaryIntegratorEngine_OpenCL::Initialize: Initializing the multipole to local translator. ";
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
        kfmout<<"KFMElectrostaticBoundaryIntegratorEngine_OpenCL::Initialize: Initializing the local to local translator. ";
    }

    fL2LConverter->SetNumberOfTermsInSeries(fNTerms);
    fL2LConverter->SetDivisions(fDivisions);
    fL2LConverter->Initialize();

    if(fVerbosity > 2)
    {
        kfmout<<"Done."<<kfmendl;
    }

    AssociateElementsAndNodes();
    InitializeMultipoleMoments();
    InitializeLocalCoefficients();
}


void
KFMElectrostaticBoundaryIntegratorEngine_OpenCL::MapField()
{
    ResetMultipoleMoments();
    ResetLocalCoefficients();
    ComputeMultipoleMoments();
    ComputeLocalCoefficients();
}

void
KFMElectrostaticBoundaryIntegratorEngine_OpenCL::AssociateElementsAndNodes()
{
    fElementNodeAssociator->Clear();
    fTree->ApplyRecursiveAction(fElementNodeAssociator);

    fMultipoleDistributor->SetElementIDList( fElementNodeAssociator->GetElementIDList() );
    fMultipoleDistributor->SetNodeList( fElementNodeAssociator->GetNodeList() );
    fMultipoleDistributor->SetOriginList( fElementNodeAssociator->GetOriginList() );

    if(fVerbosity > 2)
    {
        kfmout<<"KFMElectrostaticBoundaryIntegratorEngine_OpenCL::AssociateElementsAndNodes: Done making element to node association. "<<kfmendl;
    }
}

void
KFMElectrostaticBoundaryIntegratorEngine_OpenCL::InitializeMultipoleMoments()
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
KFMElectrostaticBoundaryIntegratorEngine_OpenCL::ResetMultipoleMoments()
{
    //reset all pre-existing multipole expansions
    fTree->ApplyCorecursiveAction(fMultipoleResetter);

    if(fVerbosity > 3)
    {
        kfmout<<"KFMElectrostaticTreeManager::ResetMultipoleMoments: Done reseting pre-exisiting multipole moments."<<kfmendl;
    }

}

void
KFMElectrostaticBoundaryIntegratorEngine_OpenCL::ComputeMultipoleMoments()
{
    //compute the individual multipole moments of each node due to owned electrodes
    fMultipoleDistributor->ProcessAndDistributeMoments();

    if(fVerbosity > 3)
    {
        kfmout<<"KFMElectrostaticBoundaryIntegratorEngine_OpenCL::ComputeMultipoleMoments: Done processing and distributing boundary element moments."<<kfmendl;
    }

    //now we perform the upward pass to collect child nodes' moments into their parents' moments
    fTree->ApplyRecursiveAction(fM2MConverter, false); //false indicates this visitation proceeds from child to parent

    if(fVerbosity > 3)
    {
        kfmout<<"KFMElectrostaticBoundaryIntegratorEngine_OpenCL::ComputeMultipoleMoments: Done performing the multipole to multipole (M2M) translations."<<kfmendl;
    }
}


void
KFMElectrostaticBoundaryIntegratorEngine_OpenCL::InitializeLocalCoefficients()
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
        kfmout<<"KFMElectrostaticBoundaryIntegratorEngine_OpenCL::InitializeLocalCoefficients: Done initializing local coefficient expansions."<<kfmendl;
    }
}

void
KFMElectrostaticBoundaryIntegratorEngine_OpenCL::ResetLocalCoefficients()
{
    //reset all existing local coefficient expansions to zero
    fTree->ApplyCorecursiveAction(fLocalCoeffResetter);

    if(fVerbosity > 3)
    {
        kfmout<<"KFMElectrostaticTreeManager::ResetLocalCoefficients: Done resetting local coefficients."<<kfmendl;
    }
}

void
KFMElectrostaticBoundaryIntegratorEngine_OpenCL::ComputeLocalCoefficients()
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
        kfmout<<"KFMElectrostaticBoundaryIntegratorEngine_OpenCL::ComputeLocalCoefficients: Done performing the multipole to local (M2L) translations."<<kfmendl;
    }

    //now form the downward distributions of the local coefficients
    fTree->ApplyRecursiveAction(fL2LConverter);

    if(fVerbosity > 3)
    {
        kfmout<<"KFMElectrostaticBoundaryIntegratorEngine_OpenCL::ComputeLocalCoefficients: Done performing the local to local (L2L) translations."<<kfmendl;
    }
}


}
