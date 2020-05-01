#include "KFMElectrostaticFieldMapper_SingleThread.hh"

#include "KFMElectrostaticMultipoleBatchCalculator.hh"
#include "KFMEmptyIdentitySetRemover.hh"
#include "KFMLeafConditionActor.hh"
#include "KFMNodeFlagValueInspector.hh"

namespace KEMField
{

KFMElectrostaticFieldMapper_SingleThread::KFMElectrostaticFieldMapper_SingleThread()
{
    fTree = nullptr;
    fContainer = nullptr;

    fDegree = 0;
    fTopLevelDivisions = 0;
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

    fM2MConverter = new KFMElectrostaticRemoteToRemoteConverter();

    //fM2LConverter = new KFMElectrostaticRemoteToLocalConverter();
    fM2LConverterInterface = new KFMElectrostaticRemoteToLocalConverterInterface();

    fL2LConverter = new KFMElectrostaticLocalToLocalConverter();
}

KFMElectrostaticFieldMapper_SingleThread::~KFMElectrostaticFieldMapper_SingleThread()
{
    delete fBatchCalc;
    delete fMultipoleDistributor;
    delete fElementNodeAssociator;
    delete fLocalCoeffInitializer;
    delete fMultipoleInitializer;
    delete fM2MConverter;

    delete fM2LConverterInterface;
    //delete fM2LConverter;

    delete fL2LConverter;
}

void KFMElectrostaticFieldMapper_SingleThread::SetTree(KFMElectrostaticTree* tree)
{
    fTree = tree;
    SetParameters(tree->GetParameters());
    KFMCube<KFMELECTROSTATICS_DIM>* world_cube;
    world_cube = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCube<KFMELECTROSTATICS_DIM>>::GetNodeObject(
        tree->GetRootNode());
    fWorldLength = world_cube->GetLength();
}

//set parameters
void KFMElectrostaticFieldMapper_SingleThread::SetParameters(KFMElectrostaticParameters params)
{
    fDegree = params.degree;
    fNTerms = (fDegree + 1) * (fDegree + 1);
    fTopLevelDivisions = params.top_level_divisions;
    fDivisions = params.divisions;
    fZeroMaskSize = params.zeromask;
    fMaximumTreeDepth = params.maximum_tree_depth;
    fVerbosity = params.verbosity;

    if (fVerbosity > 2) {
        //print the parameters
        kfmout << "KFMElectrostaticFieldMapper_SingleThread::SetParameters: top level divisions set to "
               << params.top_level_divisions << kfmendl;
        kfmout << "KFMElectrostaticFieldMapper_SingleThread::SetParameters: divisions set to " << params.divisions
               << kfmendl;
        kfmout << "KFMElectrostaticFieldMapper_SingleThread::SetParameters: degree set to " << params.degree << kfmendl;
        kfmout << "KFMElectrostaticFieldMapper_SingleThread::SetParameters: zero mask size set to " << params.zeromask
               << kfmendl;
        kfmout << "KFMElectrostaticFieldMapper_SingleThread::SetParameters: max tree depth set to "
               << params.maximum_tree_depth << kfmendl;
    }
}


void KFMElectrostaticFieldMapper_SingleThread::Initialize()
{

    if (fVerbosity > 2) {
        kfmout
            << "KFMElectrostaticFieldMapper_SingleThread::Initialize: Initializing the element multipole moment batch calculator. ";
    }

    fBatchCalc->SetDegree(fDegree);
    fBatchCalc->SetElectrostaticElementContainer(fContainer);
    fBatchCalc->Initialize();

    if (fVerbosity > 2) {
        kfmout << "Done." << kfmendl;
    }

    fLocalCoeffInitializer->SetNumberOfTermsInSeries(fNTerms);
    fMultipoleInitializer->SetNumberOfTermsInSeries(fNTerms);

    if (fVerbosity > 2) {
        kfmout
            << "KFMElectrostaticFieldMapper_SingleThread::Initialize: Initializing the multipole to multipole translator. ";
    }

    fM2MConverter->SetNumberOfTermsInSeries(fNTerms);
    fM2MConverter->SetDivisions(fDivisions);
    fM2MConverter->Initialize();

    if (fVerbosity > 2) {
        kfmout << "Done." << kfmendl;
    }

    if (fVerbosity > 2) {
        kfmout
            << "KFMElectrostaticFieldMapper_SingleThread::Initialize: Initializing the multipole to local translator. ";
    }

    //    fM2LConverter->SetLength(fWorldLength);
    //    fM2LConverter->SetMaxTreeDepth(fMaximumTreeDepth);
    //    fM2LConverter->SetNumberOfTermsInSeries(fNTerms);
    //    fM2LConverter->SetZeroMaskSize(fZeroMaskSize);
    //    fM2LConverter->SetDivisions(fDivisions);
    //    fM2LConverter->Initialize();

    fM2LConverterInterface->SetLength(fWorldLength);
    fM2LConverterInterface->SetMaxTreeDepth(fMaximumTreeDepth);
    fM2LConverterInterface->SetNumberOfTermsInSeries(fNTerms);
    fM2LConverterInterface->SetZeroMaskSize(fZeroMaskSize);
    fM2LConverterInterface->SetDivisions(fDivisions);
    fM2LConverterInterface->SetTopLevelDivisions(fTopLevelDivisions);
    fM2LConverterInterface->Initialize();

    if (fVerbosity > 2) {
        kfmout << "Done." << kfmendl;
    }

    if (fVerbosity > 2) {
        kfmout << "KFMElectrostaticFieldMapper_SingleThread::Initialize: Initializing the local to local translator. ";
    }

    fL2LConverter->SetNumberOfTermsInSeries(fNTerms);
    fL2LConverter->SetDivisions(fDivisions);
    fL2LConverter->Initialize();

    if (fVerbosity > 2) {
        kfmout << "Done." << kfmendl;
    }
}


void KFMElectrostaticFieldMapper_SingleThread::MapField()
{
    AssociateElementsAndNodes();
    InitializeMultipoleMoments();
    InitializeLocalCoefficients();
    ComputeMultipoleMoments();
    ComputeLocalCoefficients();
    CleanUp();
}

void KFMElectrostaticFieldMapper_SingleThread::AssociateElementsAndNodes()
{
    fElementNodeAssociator->Clear();
    fTree->ApplyRecursiveAction(fElementNodeAssociator);

    fMultipoleDistributor->SetElementIDList(fElementNodeAssociator->GetElementIDList());
    fMultipoleDistributor->SetNodeList(fElementNodeAssociator->GetNodeList());
    fMultipoleDistributor->SetOriginList(fElementNodeAssociator->GetOriginList());

    if (fVerbosity > 2) {
        kfmout
            << "KFMElectrostaticFieldMapper_SingleThread::AssociateElementsAndNodes: Done making element to node association. "
            << kfmendl;
    }
}

void KFMElectrostaticFieldMapper_SingleThread::InitializeMultipoleMoments()
{
    //remove any pre-existing multipole expansions
    KFMNodeObjectRemover<KFMElectrostaticNodeObjects, KFMElectrostaticMultipoleSet> remover;
    fTree->ApplyCorecursiveAction(&remover);

    //condition for a node to have a multipole expansion is based on the non-zero multipole moment flag
    KFMNodeFlagValueInspector<KFMElectrostaticNodeObjects, KFMELECTROSTATICS_FLAGS> multipole_flag_condition;
    multipole_flag_condition.SetFlagIndex(1);
    multipole_flag_condition.SetFlagValue(1);

    //now we constuct the conditional actor
    KFMConditionalActor<KFMElectrostaticNode> conditional_actor;

    conditional_actor.SetInspectingActor(&multipole_flag_condition);
    conditional_actor.SetOperationalActor(fMultipoleInitializer);

    //initialize multipole expansions for appropriate nodes
    fTree->ApplyRecursiveAction(&conditional_actor);
}

void KFMElectrostaticFieldMapper_SingleThread::ComputeMultipoleMoments()
{
    //compute the individual multipole moments of each node due to owned electrodes
    fMultipoleDistributor->ProcessAndDistributeMoments();

    if (fVerbosity > 2) {
        kfmout
            << "KFMElectrostaticFieldMapper_SingleThread::ComputeMultipoleMoments: Done processing and distributing boundary element moments."
            << kfmendl;
    }

    //now we perform the upward pass to collect child nodes' moments into their parents' moments
    fTree->ApplyRecursiveAction(fM2MConverter, false);  //false indicates this visitation proceeds from child to parent

    if (fVerbosity > 2) {
        kfmout
            << "KFMElectrostaticFieldMapper_SingleThread::ComputeMultipoleMoments: Done performing the multipole to multipole (M2M) translations."
            << kfmendl;
    }
}


void KFMElectrostaticFieldMapper_SingleThread::InitializeLocalCoefficients()  //full initialization for all nodes
{
    //delete all pre-existing local coefficient expansions
    KFMNodeObjectRemover<KFMElectrostaticNodeObjects, KFMElectrostaticLocalCoefficientSet> remover;
    fTree->ApplyCorecursiveAction(&remover);

    //initialize all of local coefficient expansions for every node
    fTree->ApplyCorecursiveAction(fLocalCoeffInitializer);

    if (fVerbosity > 2) {
        kfmout
            << "KFMElectrostaticFieldMapper_SingleThread::InitializeLocalCoefficients: Done initializing local coefficient expansions."
            << kfmendl;
    }
}

void KFMElectrostaticFieldMapper_SingleThread::ComputeLocalCoefficients()
{
    //    //compute the local coefficients due to neighbors at the same tree level
    //    fM2LConverter->Prepare(fTree);
    //    do
    //    {
    //        fTree->ApplyCorecursiveAction(fM2LConverter);
    //    }
    //    while( !(fM2LConverter->IsFinished()) );
    //    fM2LConverter->Finalize(fTree);

    //compute the local coefficients due to neighbors at the same tree level
    fM2LConverterInterface->Prepare();
    do {
        fTree->ApplyCorecursiveAction(fM2LConverterInterface);
    } while (!(fM2LConverterInterface->IsFinished()));
    fM2LConverterInterface->Finalize();

    if (fVerbosity > 2) {
        kfmout
            << "KFMElectrostaticFieldMapper_SingleThread::ComputeLocalCoefficients: Done performing the multipole to local (M2L) translations."
            << kfmendl;
    }

    //now form the downward distributions of the local coefficients
    fTree->ApplyRecursiveAction(fL2LConverter);

    if (fVerbosity > 2) {
        kfmout
            << "KFMElectrostaticFieldMapper_SingleThread::ComputeLocalCoefficients: Done performing the local to local (L2L) translations."
            << kfmendl;
    }
}

void KFMElectrostaticFieldMapper_SingleThread::CleanUp()
{
    //reset the node's ptr to the element container to null, since we will delete it
    KFMNodeObjectNullifier<KFMElectrostaticNodeObjects, KFMElectrostaticElementContainerBase<3, 1>>
        elementContainerNullifier;
    fTree->ApplyCorecursiveAction(&elementContainerNullifier);

    //now we can clean up the node objects
    //remove empty id and external id sets
    KFMEmptyIdentitySetRemover<KFMElectrostaticNodeObjects> empty_id_remover;
    fTree->ApplyCorecursiveAction(&empty_id_remover);

    //remove non-leaf local coefficients (not needed)
    KFMNodeObjectRemover<KFMElectrostaticNodeObjects, KFMElectrostaticLocalCoefficientSet> lc_remover;
    KFMLeafConditionActor<KFMElectrostaticNode> leaf_condition;
    leaf_condition.SetFalseOnLeafNodes();
    KFMConditionalActor<KFMElectrostaticNode> conditional_lc_remover;
    conditional_lc_remover.SetInspectingActor(&leaf_condition);
    conditional_lc_remover.SetOperationalActor(&lc_remover);
    fTree->ApplyRecursiveAction(&conditional_lc_remover);

    //remove multipole moments (not needed)
    KFMNodeObjectRemover<KFMElectrostaticNodeObjects, KFMElectrostaticMultipoleSet> remover;
    fTree->ApplyCorecursiveAction(&remover);
}


}  // namespace KEMField
