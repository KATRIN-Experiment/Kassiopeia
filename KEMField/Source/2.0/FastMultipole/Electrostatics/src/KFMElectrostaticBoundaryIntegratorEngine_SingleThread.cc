#include "KFMElectrostaticBoundaryIntegratorEngine_SingleThread.hh"

#include "KFMElectrostaticMultipoleBatchCalculator.hh"
#include "KFMNodeFlagValueInspector.hh"
#include "KFMEmptyIdentitySetRemover.hh"
#include "KFMLeafConditionActor.hh"

namespace KEMField
{

KFMElectrostaticBoundaryIntegratorEngine_SingleThread::KFMElectrostaticBoundaryIntegratorEngine_SingleThread()
{
    fTree = NULL;
    fContainer = NULL;

    fDegree = 0;
    fDivisions = 0;
    fZeroMaskSize = 0;
    fMaximumTreeDepth = 0;
    fVerbosity = 0;

    fBatchCalc = new KFMElectrostaticMultipoleBatchCalculator();
    fMultipoleDistributor = new KFMElectrostaticElementMultipoleDistributor();
    fMultipoleDistributor->SetBatchCalculator(fBatchCalc);
    fElementNodeAssociator = new KFMElectrostaticElementNodeAssociator();

    fLocalCoeffInitializer = new KFMElectrostaticLocalCoefficientInitializer();
    fMultipoleInitializer = new KFMElectrostaticMultipoleInitializer();

    fLocalCoeffResetter = new KFMElectrostaticLocalCoefficientResetter();
    fMultipoleResetter = new KFMElectrostaticMultipoleResetter();

    fM2MConverter = new KFMElectrostaticRemoteToRemoteConverter();
    fM2LConverter = new KFMElectrostaticRemoteToLocalConverter();
    fL2LConverter = new KFMElectrostaticLocalToLocalConverter();
};

KFMElectrostaticBoundaryIntegratorEngine_SingleThread::~KFMElectrostaticBoundaryIntegratorEngine_SingleThread()
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
KFMElectrostaticBoundaryIntegratorEngine_SingleThread::SetTree(KFMElectrostaticTree* tree)
{
    fTree = tree;
    SetParameters(tree->GetParameters());
    KFMCube<KFMELECTROSTATICS_DIM>* world_cube;
    world_cube =  KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCube<KFMELECTROSTATICS_DIM> >::GetNodeObject(tree->GetRootNode());
    fWorldLength = world_cube->GetLength();
};

//set parameters
void
KFMElectrostaticBoundaryIntegratorEngine_SingleThread::SetParameters(KFMElectrostaticParameters params)
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
        kfmout<<"KFMElectrostaticBoundaryIntegratorEngine_SingleThread::SetParameters: divisions set to "<<params.divisions<<kfmendl;
        kfmout<<"KFMElectrostaticBoundaryIntegratorEngine_SingleThread::SetParameters: degree set to "<<params.degree<<kfmendl;
        kfmout<<"KFMElectrostaticBoundaryIntegratorEngine_SingleThread::SetParameters: zero mask size set to "<<params.zeromask<<kfmendl;
        kfmout<<"KFMElectrostaticBoundaryIntegratorEngine_SingleThread::SetParameters: max tree depth set to "<<params.maximum_tree_depth<<kfmendl;
    }
}


void
KFMElectrostaticBoundaryIntegratorEngine_SingleThread::Initialize()
{

    if(fVerbosity > 2)
    {
        kfmout<<"KFMElectrostaticBoundaryIntegratorEngine_SingleThread::Initialize: Initializing the element multipole moment batch calculator. ";
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
        kfmout<<"KFMElectrostaticBoundaryIntegratorEngine_SingleThread::Initialize: Initializing the multipole to multipole translator. ";
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
        kfmout<<"KFMElectrostaticBoundaryIntegratorEngine_SingleThread::Initialize: Initializing the multipole to local translator. ";
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
        kfmout<<"KFMElectrostaticBoundaryIntegratorEngine_SingleThread::Initialize: Initializing the local to local translator. ";
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
KFMElectrostaticBoundaryIntegratorEngine_SingleThread::MapField()
{
    ResetMultipoleMoments();
    ResetLocalCoefficients();
    ComputeMultipoleMoments();
    ComputeLocalCoefficients();
}

void
KFMElectrostaticBoundaryIntegratorEngine_SingleThread::AssociateElementsAndNodes()
{
    fElementNodeAssociator->Clear();
    fTree->ApplyRecursiveAction(fElementNodeAssociator);

    fMultipoleDistributor->SetElementIDList( fElementNodeAssociator->GetElementIDList() );
    fMultipoleDistributor->SetNodeList( fElementNodeAssociator->GetNodeList() );
    fMultipoleDistributor->SetOriginList( fElementNodeAssociator->GetOriginList() );

    if(fVerbosity > 2)
    {
        kfmout<<"KFMElectrostaticBoundaryIntegratorEngine_SingleThread::AssociateElementsAndNodes: Done making element to node association. "<<kfmendl;
    }
}

void
KFMElectrostaticBoundaryIntegratorEngine_SingleThread::InitializeMultipoleMoments()
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
KFMElectrostaticBoundaryIntegratorEngine_SingleThread::ResetMultipoleMoments()
{
    //reset all pre-existing multipole expansions
    fTree->ApplyCorecursiveAction(fMultipoleResetter);

    if(fVerbosity > 3)
    {
        kfmout<<"KFMElectrostaticTreeManager::ResetMultipoleMoments: Done reseting pre-exisiting multipole moments."<<kfmendl;
    }

}

void
KFMElectrostaticBoundaryIntegratorEngine_SingleThread::ComputeMultipoleMoments()
{
    //compute the individual multipole moments of each node due to owned electrodes
    fMultipoleDistributor->ProcessAndDistributeMoments();

    if(fVerbosity > 3)
    {
        kfmout<<"KFMElectrostaticBoundaryIntegratorEngine_SingleThread::ComputeMultipoleMoments: Done processing and distributing boundary element moments."<<kfmendl;
    }

    //now we perform the upward pass to collect child nodes' moments into their parents' moments
    fTree->ApplyRecursiveAction(fM2MConverter, false); //false indicates this visitation proceeds from child to parent

    if(fVerbosity > 3)
    {
        kfmout<<"KFMElectrostaticBoundaryIntegratorEngine_SingleThread::ComputeMultipoleMoments: Done performing the multipole to multipole (M2M) translations."<<kfmendl;
    }
}


void
KFMElectrostaticBoundaryIntegratorEngine_SingleThread::InitializeLocalCoefficients()
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
        kfmout<<"KFMElectrostaticBoundaryIntegratorEngine_SingleThread::InitializeLocalCoefficients: Done initializing local coefficient expansions."<<kfmendl;
    }
}

void
KFMElectrostaticBoundaryIntegratorEngine_SingleThread::ResetLocalCoefficients()
{
    //reset all existing local coefficient expansions to zero
    fTree->ApplyCorecursiveAction(fLocalCoeffResetter);

    if(fVerbosity > 3)
    {
        kfmout<<"KFMElectrostaticTreeManager::ResetLocalCoefficients: Done resetting local coefficients."<<kfmendl;
    }
}

void
KFMElectrostaticBoundaryIntegratorEngine_SingleThread::ComputeLocalCoefficients()
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
        kfmout<<"KFMElectrostaticBoundaryIntegratorEngine_SingleThread::ComputeLocalCoefficients: Done performing the multipole to local (M2L) translations."<<kfmendl;
    }

    //now form the downward distributions of the local coefficients
    fTree->ApplyRecursiveAction(fL2LConverter);

    if(fVerbosity > 3)
    {
        kfmout<<"KFMElectrostaticBoundaryIntegratorEngine_SingleThread::ComputeLocalCoefficients: Done performing the local to local (L2L) translations."<<kfmendl;
    }
}


}
