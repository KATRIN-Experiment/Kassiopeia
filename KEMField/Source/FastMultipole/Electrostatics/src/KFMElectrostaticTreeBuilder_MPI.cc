#include "KFMElectrostaticTreeBuilder_MPI.hh"

#include "KFMCompoundInspectingActor.hh"
#include "KFMCubicSpaceTreeStaticLoadBalancer.hh"
#include "KFMElectrostaticNodeWorkScoreCalculator.hh"
#include "KFMLeafConditionActor.hh"
#include "KFMLevelConditionActor.hh"
#include "KFMNodeChildToParentFlagValueInitializer.hh"
#include "KFMNodeFlagValueInitializer.hh"
#include "KFMNodeFlagValueInspector.hh"
#include "KFMNodeParentToChildFlagValueInspector.hh"
#include "KFMNonEmptyIdentitySetActor.hh"
#include "KFMSubdivisionConditionAggressive.hh"
#include "KFMSubdivisionConditionBalanced.hh"
#include "KFMSubdivisionConditionGuided.hh"

#define MPI_SINGLE_PROCESS if (KMPIInterface::GetInstance()->GetProcess() == 0)

#define NODELIST_SIZE_TAG   800
#define NODELIST_ASSIGN_TAG 801

namespace KEMField
{

//extracted electrode data
void KFMElectrostaticTreeBuilder_MPI::SetElectrostaticElementContainer(
    KFMElectrostaticElementContainerBase<KFMELECTROSTATICS_DIM, KFMELECTROSTATICS_BASIS>* container)
{
    fContainer = container;
};

////////////////////////////////////////////////////////////////////////////////


KFMElectrostaticElementContainerBase<KFMELECTROSTATICS_DIM, KFMELECTROSTATICS_BASIS>*
KFMElectrostaticTreeBuilder_MPI::GetElectrostaticElementContainer()
{
    return fContainer;
};

////////////////////////////////////////////////////////////////////////////////


void KFMElectrostaticTreeBuilder_MPI::SetTree(KFMElectrostaticTree* tree)
{
    fTree = tree;
    KFMElectrostaticParameters params = tree->GetParameters();

    fDegree = params.degree;
    fNTerms = (fDegree + 1) * (fDegree + 1);
    fTopLevelDivisions = params.top_level_divisions;
    fDivisions = params.divisions;
    fZeroMaskSize = params.zeromask;
    fMaximumTreeDepth = params.maximum_tree_depth;
    fInsertionRatio = params.insertion_ratio;
    fRegionSizeFactor = std::fabs(params.region_expansion_factor);
    fVerbosity = params.verbosity;

    if (!(params.use_region_estimation)) {
        fUseRegionEstimation = false;
        fWorldCenter[0] = params.world_center_x;
        fWorldCenter[1] = params.world_center_y;
        fWorldCenter[2] = params.world_center_z;
        fWorldLength = params.world_length;
    }
    else {
        fUseRegionEstimation = true;
    }

    if (fVerbosity > 1) {
        MPI_SINGLE_PROCESS
        {
            //print the parameters
            kfmout << "KFMElectrostaticTreeBuilder_MPI::SetParameters: top level divisions set to "
                   << params.top_level_divisions << kfmendl;
            kfmout << "KFMElectrostaticTreeBuilder_MPI::SetParameters: divisions set to " << params.divisions
                   << kfmendl;
            kfmout << "KFMElectrostaticTreeBuilder_MPI::SetParameters: degree set to " << params.degree << kfmendl;
            kfmout << "KFMElectrostaticTreeBuilder_MPI::SetParameters: zero mask size set to " << params.zeromask
                   << kfmendl;
            kfmout << "KFMElectrostaticTreeBuilder_MPI::SetParameters: max tree depth set to "
                   << params.maximum_tree_depth << kfmendl;

            if (!(params.use_region_estimation)) {
                kfmout << "KFMElectrostaticTreeBuilder_MPI::SetParameters: using user defined world cube volume"
                       << kfmendl;
                kfmout << "KFMElectrostaticTreeBuilder_MPI::SetParameters: world cube center set to ("
                       << params.world_center_x << ", " << params.world_center_y << ", " << params.world_center_z
                       << ") " << kfmendl;
                kfmout << "KFMElectrostaticTreeBuilder_MPI::SetParameters: world cube side length set to ("
                       << params.world_length << kfmendl;
            }
            else {
                kfmout << "KFMElectrostaticTreeBuilder_MPI::SetParameters: using region size estimation" << kfmendl;
                kfmout << "KFMElectrostaticTreeBuilder_MPI::SetParameters: region expansion factor set to "
                       << params.region_expansion_factor << kfmendl;
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////


KFMElectrostaticTree* KFMElectrostaticTreeBuilder_MPI::GetTree()
{
    return fTree;
};

////////////////////////////////////////////////////////////////////////////////


void KFMElectrostaticTreeBuilder_MPI::ConstructRootNode()
{
    fTree->RestrictActionBehavior(false);

    if (fUseRegionEstimation) {
        if (fVerbosity > 2) {
            MPI_SINGLE_PROCESS
            {
                kfmout << "KFMElectrostaticTreeBuilder_MPI::Initialize: Region size estimator running. " << kfmendl;
            }
        }

        //the region size estimator
        KFMElectrostaticRegionSizeEstimator regionSizeEstimator;

        regionSizeEstimator.SetElectrostaticElementContainer(fContainer);
        regionSizeEstimator.ComputeEstimate();
        KFMCube<3> estimate = regionSizeEstimator.GetCubeEstimate();

        fWorldCenter = estimate.GetCenter();
        fWorldLength = fRegionSizeFactor * (estimate.GetLength());

        if (fVerbosity > 2) {
            MPI_SINGLE_PROCESS
            {
                kfmout << "KFMElectrostaticTreeBuilder_MPI::Initialize: Estimated world cube center is ("
                       << fWorldCenter[0] << ", " << fWorldCenter[1] << ", " << fWorldCenter[2] << ")" << kfmendl;
                kfmout << "KFMElectrostaticTreeBuilder_MPI::Initialize: Estimated world cube size length is "
                       << fWorldLength << kfmendl;
            }
        }
    }

    KFMCube<3>* world_volume;
    world_volume = new KFMCube<3>(fWorldCenter, fWorldLength);
    KFMPoint<3> center = world_volume->GetCenter();

    unsigned int n_elements = fContainer->GetNElements();

    if (fVerbosity > 2) {
        MPI_SINGLE_PROCESS
        {
            kfmout << "KFMElectrostaticTreeBuilder_MPI::ConstructRootNode: Constructing root node with center at ("
                   << center[0] << ", " << center[1] << ", " << center[2] << ")." << kfmendl;
            kfmout << "KFMElectrostaticTreeBuilder_MPI::ConstructRootNode: Root node has side length of "
                   << fWorldLength << " and contains " << n_elements << " boundary elements." << kfmendl;
        }
    }

    KFMCubicSpaceTreeProperties<3>* tree_prop = fTree->GetTreeProperties();

    tree_prop->SetMaxTreeDepth(fMaximumTreeDepth);
    tree_prop->SetCubicNeighborOrder(fZeroMaskSize);
    unsigned int dim[3] = {(unsigned int) fDivisions, (unsigned int) fDivisions, (unsigned int) fDivisions};
    unsigned int top_level_dim[3] = {(unsigned int) fTopLevelDivisions,
                                     (unsigned int) fTopLevelDivisions,
                                     (unsigned int) fTopLevelDivisions};
    tree_prop->SetDimensions(dim);
    tree_prop->SetTopLevelDimensions(top_level_dim);

    KFMElectrostaticNode* root = fTree->GetRootNode();

    //set the world volume
    KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCube<3>>::SetNodeObject(world_volume, root);

    //set the tree properties
    KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCubicSpaceTreeProperties<3>>::SetNodeObject(tree_prop, root);

    //set the element container
    KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMElectrostaticElementContainerBase<3, 1>>::SetNodeObject(
        fContainer,
        root);

    //add the complete id set of all elements to the root node
    KFMIdentitySet* root_list = new KFMIdentitySet();
    for (unsigned int i = 0; i < n_elements; i++) {
        root_list->AddID(i);
    }

    //set the id set
    KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMIdentitySet>::SetNodeObject(root_list, root);

    //set basis node properties
    root->SetID(tree_prop->RegisterNode());
    root->SetIndex(0);
    root->SetParent(NULL);
}

////////////////////////////////////////////////////////////////////////////////


void KFMElectrostaticTreeBuilder_MPI::PerformSpatialSubdivision()
{
    fTree->RestrictActionBehavior(false);

    //conditions for subdivision of a node
    KFMInsertionCondition<3> basic_insertion_condition;
    basic_insertion_condition.SetInsertionRatio(fInsertionRatio);

    if (fSubdivisionCondition == NULL) {
        //subdivision condition was unset, so we default to aggressive
        //since it is the only one which takes no paramters
        fSubdivisionCondition =
            new KFMSubdivisionConditionAggressive<KFMELECTROSTATICS_DIM, KFMElectrostaticNodeObjects>();
        fSubdivisionConditionIsOwned = true;
    }

    fSubdivisionCondition->SetInsertionCondition(&basic_insertion_condition);
    fTree->SetSubdivisionCondition(fSubdivisionCondition);

    //things to do on a node after it has been visited by the progenitor
    KFMCubicSpaceBallSorter<3, KFMElectrostaticNodeObjects> bball_sorter;
    bball_sorter.SetInsertionCondition(&basic_insertion_condition);
    fTree->AddPostSubdivisionAction(&bball_sorter);

    KFMObjectContainer<KFMBall<3>>* bballs = fContainer->GetBoundingBallContainer();
    fSubdivisionCondition->SetBoundingBallContainer(bballs);
    bball_sorter.SetBoundingBallContainer(bballs);

    if (fVerbosity > 2) {
        MPI_SINGLE_PROCESS
        {
            kfmout << "KFMElectrostaticTreeBuilder_MPI::PerformSpatialSubdivision: Subdividing region using the "
                   << fSubdivisionCondition->Name() << " strategy " << kfmendl;
        }
    }

    fTree->ConstructTree();

    if (fVerbosity > 2) {
        MPI_SINGLE_PROCESS
        {
            kfmout
                << "KFMElectrostaticTreeBuilder_MPI::PerformSpatialSubdivision: Done performing spatial subdivision. "
                << kfmendl;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////


void KFMElectrostaticTreeBuilder_MPI::FlagNonZeroMultipoleNodes()
{
    fTree->RestrictActionBehavior(false);

    //the condition (non-empty id set)
    KFMNonEmptyIdentitySetActor<KFMElectrostaticNodeObjects> non_empty_condition;

    //the flag value initializer
    KFMNodeChildToParentFlagValueInitializer<KFMElectrostaticNodeObjects, KFMELECTROSTATICS_FLAGS> flag_init;
    flag_init.SetFlagIndex(1);
    flag_init.SetFlagValue(1);

    KFMConditionalActor<KFMElectrostaticNode> conditional_actor;

    conditional_actor.SetInspectingActor(&non_empty_condition);
    conditional_actor.SetOperationalActor(&flag_init);

    fTree->ApplyRecursiveAction(&conditional_actor);
}

////////////////////////////////////////////////////////////////////////////////


void KFMElectrostaticTreeBuilder_MPI::PerformAdjacencySubdivision()
{
    fTree->RestrictActionBehavior(false);

    //adjacency progenation condition
    KFMElectrostaticAdjacencyProgenitor adjacencyProgenitor;
    adjacencyProgenitor.SetZeroMaskSize(fZeroMaskSize);

    //the condition that we use is that the node's adjacent to a node which
    //has non-zero multipole moments, and is not a leaf node

    //leaf condition inspector
    KFMLeafConditionActor<KFMElectrostaticNode> leaf_condition;
    leaf_condition.SetFalseOnLeafNodes();

    //the non-zero multipole flag inspector
    KFMNodeFlagValueInspector<KFMElectrostaticNodeObjects, KFMELECTROSTATICS_FLAGS> multipole_flag_condition;
    multipole_flag_condition.SetFlagIndex(1);
    multipole_flag_condition.SetFlagValue(1);

    //non-zero multipole flag inspector for all of the node's children
    KFMNodeParentToChildFlagValueInspector<KFMElectrostaticNodeObjects, KFMELECTROSTATICS_FLAGS>
        child_multipole_flag_condition;
    child_multipole_flag_condition.SetFlagIndex(1);
    child_multipole_flag_condition.SetFlagValue(1);
    child_multipole_flag_condition.UseOrCondition();

    //compound condition
    KFMCompoundInspectingActor<KFMElectrostaticNode> compound_inspector;
    compound_inspector.UseAndCondition();

    compound_inspector.AddInspectingActor(&leaf_condition);
    compound_inspector.AddInspectingActor(&multipole_flag_condition);
    compound_inspector.AddInspectingActor(&child_multipole_flag_condition);

    //now we constuct the conditional actor
    KFMConditionalActor<KFMElectrostaticNode> conditional_actor;

    conditional_actor.SetInspectingActor(&compound_inspector);
    conditional_actor.SetOperationalActor(&adjacencyProgenitor);

    fTree->ApplyCorecursiveAction(&conditional_actor);

    if (fVerbosity > 3) {
        MPI_SINGLE_PROCESS
        {
            kfmout
                << "KFMElectrostaticTreeBuilder_MPI::PerformAdjacencySubdivision: Done performing subdivision of nodes satisfying adjacency condition."
                << kfmendl;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////


void KFMElectrostaticTreeBuilder_MPI::FlagPrimaryNodes()
{
    fTree->RestrictActionBehavior(false);

    //for charge density solving we need to flag nodes which contain element centroids
    //(these get the 'primary' status flag)
    KFMElectrostaticTreeNavigator navigator;

    //the flag value initializer
    KFMNodeChildToParentFlagValueInitializer<KFMElectrostaticNodeObjects, KFMELECTROSTATICS_FLAGS> flag_init;
    flag_init.SetFlagIndex(0);
    flag_init.SetFlagValue(1);

    unsigned int n_elem = fContainer->GetNElements();
    //loop over all elements of surface container
    for (unsigned int i = 0; i < n_elem; i++) {
        KFMPoint<3> centroid = *(fContainer->GetCentroid(i));
        navigator.SetPoint(&centroid);
        navigator.ApplyAction(fTree->GetRootNode());

        if (navigator.Found()) {
            KFMElectrostaticNode* leaf_node = navigator.GetLeafNode();
            flag_init.ApplyAction(leaf_node);
        }
        else {
            kfmout
                << "KFMElectrostaticBoundaryIntegrator::FlagPrimaryNodes: Error, element centroid not found in region."
                << kfmendl;
            KMPIInterface::GetInstance()->Finalize();
            kfmexit(1);
        }
    }

    if (fVerbosity > 2) {
        MPI_SINGLE_PROCESS
        {
            kfmout << "KFMElectrostaticTreeBuilder_MPI::FlagPrimaryNodes: Done flagging primary nodes." << kfmendl;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

void KFMElectrostaticTreeBuilder_MPI::DetermineSourceNodes()
{
    //get partner process and number of processes
    unsigned int n_processes = 0;
    unsigned int partner = 0;

    if (KMPIInterface::GetInstance()->SplitMode()) {
        n_processes = KMPIInterface::GetInstance()->GetNSubGroupProcesses();
        partner = KMPIInterface::GetInstance()->GetPartnerProcessID();
    }
    else {
        n_processes = KMPIInterface::GetInstance()->GetNProcesses();
    }

    //condition for a node to have a multipole expansion is based on the non-zero multipole moment flag
    KFMNodeFlagValueInspector<KFMElectrostaticNodeObjects, KFMELECTROSTATICS_FLAGS> multipole_flag_condition;
    multipole_flag_condition.SetFlagIndex(1);
    multipole_flag_condition.SetFlagValue(1);

    //determine the number of source nodes at the top level of the tree
    unsigned int n_children = fTree->GetRootNode()->GetNChildren();
    unsigned int n_top_level_source_nodes = 0;
    std::vector<unsigned int> source_node_indexes;

    for (unsigned int i = 0; i < n_children; i++) {
        if (multipole_flag_condition.ConditionIsSatisfied(fTree->GetRootNode()->GetChild(i))) {
            source_node_indexes.push_back(i);
            n_top_level_source_nodes++;
        }
    }

    //score calculator
    KFMElectrostaticNodeWorkScoreCalculator score_calc;
    score_calc.SetFFTWeight(fFFTWeight);

    //seem to get better balancing by using zero weight for sparse matrix work...need to study this further
    //score_calc.SetSparseMatrixWeight(fSparseMatrixWeight);
    score_calc.SetSparseMatrixWeight(0.0);

    if (fSubdivisionCondition->Name() == std::string("balanced")) {
        KFMSubdivisionConditionBalanced<KFMELECTROSTATICS_DIM, KFMElectrostaticNodeObjects>* sbdc = NULL;
        sbdc = dynamic_cast<KFMSubdivisionConditionBalanced<KFMELECTROSTATICS_DIM, KFMElectrostaticNodeObjects>*>(
            fSubdivisionCondition);
        double bd = sbdc->GetBiasDegree();
        score_calc.SetNTerms((bd + 1.0) * (bd + 2.0) / 2.0);
    }
    else {
    }
    score_calc.SetDivisions(fDivisions);
    score_calc.SetZeroMaskSize(fZeroMaskSize);

    //now we construct a collection of all of the source nodes and their info
    std::vector<KFMWorkBlock<KFMELECTROSTATICS_DIM>> allBlocks;
    double block_length = fWorldLength / (double) fTopLevelDivisions;
    for (unsigned int i = 0; i < n_top_level_source_nodes; i++) {
        KFMWorkBlock<KFMELECTROSTATICS_DIM> block;
        block.index = source_node_indexes[i];
        KFMElectrostaticNode* node = fTree->GetRootNode()->GetChild(source_node_indexes[i]);
        score_calc.ApplyAction(node);
        block.score = score_calc.GetNodeScore();
        KFMPoint<KFMELECTROSTATICS_DIM> center =
            KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCube<KFMELECTROSTATICS_DIM>>::GetNodeObject(node)
                ->GetCenter();
        for (unsigned int i = 0; i < KFMELECTROSTATICS_DIM; i++) {
            block.spatial_coordinates[i] = center[i];
        }
        allBlocks.push_back(block);
    }

    //load balancing calculator
    std::vector<unsigned int> process_block_ids;
    if (KMPIInterface::GetInstance()->SplitMode()) {
        if (KMPIInterface::GetInstance()->IsEvenGroupMember()) {
            KFMCubicSpaceTreeStaticLoadBalancer<KFMELECTROSTATICS_DIM> load_balancer;
            load_balancer.SetMaxIterations(10 * allBlocks.size() * n_processes);
            load_balancer.SetVerbosity(fVerbosity);
            load_balancer.SetBlockLength(block_length);
            load_balancer.SetDivisions(fTopLevelDivisions);
            load_balancer.SetNeighborOrder(fZeroMaskSize);
            load_balancer.SetAlpha(1.0);  //weight given to scores
            load_balancer.SetBeta(0.0);   //weight given to region compactness
            load_balancer.SetGamma(0.9);  //rate at which temperature declines
            load_balancer.SetBlocks(&allBlocks);
            load_balancer.Initialize();
            load_balancer.EstimateBalancedWork();

            //get the list of blocks assigned to this process
            load_balancer.GetProcessBlockIdentities(&process_block_ids);

            //send the list of blocks assigned to this process to our partner process
            unsigned int n_block_ids = process_block_ids.size();
            int ret_val = MPI_Send(&n_block_ids, 1, MPI_UNSIGNED, partner, NODELIST_SIZE_TAG, MPI_COMM_WORLD);

            if (ret_val != MPI_SUCCESS) {
                int proc = KMPIInterface::GetInstance()->GetProcess();
                kfmout << "KFMElectrostaticTreeBuilder:: Error using MPI_Send from process: " << proc << " to "
                       << partner << kfmendl;
                KMPIInterface::GetInstance()->Finalize();
                std::exit(1);
            }

            ret_val = MPI_Send(&(process_block_ids[0]),
                               n_block_ids,
                               MPI_UNSIGNED,
                               partner,
                               NODELIST_ASSIGN_TAG,
                               MPI_COMM_WORLD);

            if (ret_val != MPI_SUCCESS) {
                int proc = KMPIInterface::GetInstance()->GetProcess();
                kfmout << "KFMElectrostaticTreeBuilder:: Error using MPI_Send from process: " << proc << " to "
                       << partner << kfmendl;
                KMPIInterface::GetInstance()->Finalize();
                std::exit(1);
            }
        }
        else {
            //we are an odd process, so recieve the node list to use from our partner
            unsigned int n_block_ids;
            MPI_Status stat;
            int ret_val = MPI_Recv(&n_block_ids,
                                   1,
                                   MPI_UNSIGNED,
                                   KMPIInterface::GetInstance()->GetPartnerProcessID(),
                                   NODELIST_SIZE_TAG,
                                   MPI_COMM_WORLD,
                                   &stat);

            if (ret_val != MPI_SUCCESS) {
                int proc = KMPIInterface::GetInstance()->GetProcess();
                kfmout << "KFMElectrostaticTreeBuilder:: Error using MPI_Recv to process: " << proc << " from "
                       << partner << kfmendl;
                KMPIInterface::GetInstance()->Finalize();
                std::exit(1);
            }

            process_block_ids.resize(n_block_ids);
            ret_val = MPI_Recv(&(process_block_ids[0]),
                               n_block_ids,
                               MPI_UNSIGNED,
                               KMPIInterface::GetInstance()->GetPartnerProcessID(),
                               NODELIST_ASSIGN_TAG,
                               MPI_COMM_WORLD,
                               &stat);

            if (ret_val != MPI_SUCCESS) {
                int proc = KMPIInterface::GetInstance()->GetProcess();
                kfmout << "KFMElectrostaticTreeBuilder:: Error using MPI_Recv to process: " << proc << " from "
                       << partner << kfmendl;
                KMPIInterface::GetInstance()->Finalize();
                std::exit(1);
            }
        }
    }
    else {
        //no split mode, all nodes do same work
        KFMCubicSpaceTreeStaticLoadBalancer<KFMELECTROSTATICS_DIM> load_balancer;
        load_balancer.SetMaxIterations(10 * allBlocks.size() * n_processes);
        load_balancer.SetVerbosity(fVerbosity);
        load_balancer.SetBlockLength(block_length);
        load_balancer.SetDivisions(fTopLevelDivisions);
        load_balancer.SetNeighborOrder(fZeroMaskSize);
        load_balancer.SetAlpha(1.0);  //weight given to scores
        load_balancer.SetBeta(0.0);   //weight given to region compactness
        load_balancer.SetGamma(0.9);  //rate at which temperature declines
        load_balancer.SetBlocks(&allBlocks);
        load_balancer.Initialize();
        load_balancer.EstimateBalancedWork();

        //get the list of blocks assigned to this process
        load_balancer.GetProcessBlockIdentities(&process_block_ids);
    }

    //now we decide which top level nodes are handled by this process
    fSourceNodeCollection.clear();
    for (unsigned int j = 0; j < process_block_ids.size(); j++) {
        fSourceNodeCollection.push_back(fTree->GetRootNode()->GetChild(process_block_ids[j]));
    }

    fNSourceNodes = fSourceNodeCollection.size();

    if (fNSourceNodes == 0) {
        //error, abort, too many processes allocated for this job
        kfmout
            << "KFMElectrostaticTreeBuilder_MPI::DetermineSourceNodes(): Error, too many MPI processes allocated for this job. ";
        kfmout << "The total number of top level source nodes is: " << n_top_level_source_nodes
               << " but the number of processes is: " << n_processes << ". ";
        kfmout
            << "Please either increase the granularity of the spatial divisions or allocate fewer processes for this job. "
            << kfmendl;
        kfmexit(1);
    }

    //send message about which source nodes are assigned to each process
    if (fVerbosity > 1) {
        std::stringstream ss;
        ss << "Process #" << KMPIInterface::GetInstance()->GetProcess()
           << " has been assigned source nodes w/ child indexes: {";
        for (unsigned int i = 0; i < fSourceNodeCollection.size() - 1; i++) {
            ss << fSourceNodeCollection[i]->GetIndex() << ", ";
        };
        ss << fSourceNodeCollection.back()->GetIndex() << "} ";
        ss << std::endl;
        KMPIInterface::GetInstance()->PrintMessage(ss.str());
    }

    //determine the source region
    fSourceVolume.Clear();
    for (unsigned int i = 0; i < fNSourceNodes; i++) {
        KFMCube<KFMELECTROSTATICS_DIM>* cube = NULL;
        cube = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCube<KFMELECTROSTATICS_DIM>>::GetNodeObject(
            fSourceNodeCollection[i]);
        fSourceVolume.AddCube(cube);
    }

    //determine the nodes that are not part of the source region
    for (unsigned int i = 0; i < n_children; i++) {
        KFMElectrostaticNode* node = fTree->GetRootNode()->GetChild(i);
        bool non_source = true;
        for (unsigned int j = 0; j < fSourceNodeCollection.size(); j++) {
            if (fSourceNodeCollection[j] == node) {
                non_source = false;
            };
        }

        if (non_source) {
            fNonSourceNodeCollection.push_back(node);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

void KFMElectrostaticTreeBuilder_MPI::GetSourceNodeIndexes(std::vector<unsigned int>* source_node_indexes) const
{
    source_node_indexes->clear();
    for (unsigned int i = 0; i < fNSourceNodes; i++) {
        source_node_indexes->push_back(fSourceNodeCollection[i]->GetIndex());
    }
}

////////////////////////////////////////////////////////////////////////////////


void KFMElectrostaticTreeBuilder_MPI::GetTargetNodeIndexes(std::vector<unsigned int>* target_node_indexes) const
{
    target_node_indexes->clear();
    for (unsigned int i = 0; i < fNTargetNodes; i++) {
        target_node_indexes->push_back(fTargetNodeCollection[i]->GetIndex());
    }
}


////////////////////////////////////////////////////////////////////////////////

void KFMElectrostaticTreeBuilder_MPI::DetermineTargetNodes()
{
    //create the neighbor finder
    KFMCubicSpaceNodeNeighborFinder<KFMELECTROSTATICS_DIM, KFMElectrostaticNodeObjects> neighbor_finder;

    //the primacy flag inspector (check if a node contains target points)
    KFMNodeFlagValueInspector<KFMElectrostaticNodeObjects, KFMELECTROSTATICS_FLAGS> primacy_inspector;
    primacy_inspector.SetFlagIndex(0);
    primacy_inspector.SetFlagValue(1);

    std::vector<KFMElectrostaticNode*> neighbors;

    //in this function we collect the neighbors of the source nodes
    fTargetNodeCollection.clear();
    for (unsigned int i = 0; i < fSourceNodeCollection.size(); i++) {
        neighbors.clear();
        neighbor_finder.GetAllNeighbors(fSourceNodeCollection[i], fZeroMaskSize, &neighbors);
        for (unsigned int j = 0; j < neighbors.size(); j++) {
            if (neighbors[j] != NULL) {
                if (primacy_inspector.ConditionIsSatisfied(neighbors[j])) {
                    bool is_present = false;
                    for (unsigned int n = 0; n < fTargetNodeCollection.size(); n++) {
                        if (fTargetNodeCollection[n] == neighbors[j]) {
                            is_present = true;
                            break;
                        }
                    }

                    if (!is_present) {
                        fTargetNodeCollection.push_back(neighbors[j]);
                    }
                }
            }
        }
    }

    fNTargetNodes = fTargetNodeCollection.size();

    //send message about which target nodes are assigned to each process
    if (fVerbosity > 4) {
        std::stringstream ss;
        ss << "Process #" << KMPIInterface::GetInstance()->GetProcess()
           << " has been assigned target nodes w/ child indexes: { ";
        for (unsigned int i = 0; i < fTargetNodeCollection.size() - 1; i++) {
            ss << fTargetNodeCollection[i]->GetIndex() << ", ";
        };
        ss << fTargetNodeCollection.back()->GetIndex() << "}";
        ss << std::endl;
        KMPIInterface::GetInstance()->PrintMessage(ss.str());
    }


    //determine the target region
    fTargetVolume.Clear();
    for (unsigned int i = 0; i < fNTargetNodes; i++) {
        KFMCube<KFMELECTROSTATICS_DIM>* cube = NULL;
        cube = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCube<KFMELECTROSTATICS_DIM>>::GetNodeObject(
            fTargetNodeCollection[i]);
        fTargetVolume.AddCube(cube);
    }


    //compute the set of nodes that are not in the target regions
    unsigned int n_children = fTree->GetRootNode()->GetNChildren();
    for (unsigned int i = 0; i < n_children; i++) {
        KFMElectrostaticNode* node = fTree->GetRootNode()->GetChild(i);
        bool non_target = true;
        for (unsigned int j = 0; j < fTargetNodeCollection.size(); j++) {
            if (fTargetNodeCollection[j] == node) {
                non_target = false;
            };
        }

        if (non_target) {
            fNonTargetNodeCollection.push_back(node);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

void KFMElectrostaticTreeBuilder_MPI::RemoveExtraneousData()
{
    //restricting actions to non-source nodes

    //now we apply an actor which removes all identity sets from nodes
    //in the non source region
    KFMNodeObjectRemover<KFMElectrostaticNodeObjects, KFMIdentitySet> id_set_remover;
    fTree->SetReducedActionCollection(&fNonSourceNodeCollection);
    fTree->RestrictActionBehavior(true);
    fTree->ApplyRecursiveAction(&id_set_remover);

    //remove unneed multipole moment sets
    KFMNodeObjectRemover<KFMElectrostaticNodeObjects, KFMElectrostaticMultipoleSet> multipole_set_remover;
    fTree->SetReducedActionCollection(&fNonSourceNodeCollection);
    fTree->RestrictActionBehavior(true);
    fTree->ApplyRecursiveAction(&multipole_set_remover);

    //remove the source flags
    KFMNodeChildToParentFlagValueInitializer<KFMElectrostaticNodeObjects, KFMELECTROSTATICS_FLAGS> source_flag_init;
    source_flag_init.SetFlagIndex(1);
    source_flag_init.SetFlagValue(0);
    fTree->SetReducedActionCollection(&fNonSourceNodeCollection);
    fTree->RestrictActionBehavior(true);
    fTree->ApplyRecursiveAction(&source_flag_init);

    //make sure that root node is still listed as being a source containing node
    KFMNodeChildToParentFlagValueInitializer<KFMElectrostaticNodeObjects, KFMELECTROSTATICS_FLAGS>
        root_source_flag_init;
    root_source_flag_init.SetFlagIndex(1);
    root_source_flag_init.SetFlagValue(1);
    root_source_flag_init.ApplyAction(fTree->GetRootNode());

    //now we restrict action to the non-target nodes

    //the node level condition
    KFMLevelConditionActor<KFMElectrostaticNode> level_condition;
    level_condition.SetLevel(1);
    level_condition.SetGreaterThan();

    //the flag value initializer
    KFMNodeFlagValueInitializer<KFMElectrostaticNodeObjects, KFMELECTROSTATICS_FLAGS> target_flag_init;
    target_flag_init.SetFlagIndex(0);
    target_flag_init.SetFlagValue(0);

    //conditional actor
    KFMConditionalActor<KFMElectrostaticNode> conditional_actor;
    conditional_actor.SetInspectingActor(&level_condition);
    conditional_actor.SetOperationalActor(&target_flag_init);

    //restrict the tree action to only visit these nodes
    fTree->SetReducedActionCollection(&fNonTargetNodeCollection);
    fTree->RestrictActionBehavior(true);
    fTree->ApplyRecursiveAction(&conditional_actor);

    //remove unneed local moment sets
    KFMNodeObjectRemover<KFMElectrostaticNodeObjects, KFMElectrostaticLocalCoefficientSet> local_set_remover;
    conditional_actor.SetOperationalActor(&local_set_remover);
    fTree->SetReducedActionCollection(&fNonTargetNodeCollection);
    fTree->RestrictActionBehavior(true);
    fTree->ApplyRecursiveAction(&conditional_actor);

    //remove all the nodes underneath those nodes
    //which are in the non-target region, as they are unneeded
    //by this MPI process

    KFMNodeObjectNullifier<KFMElectrostaticNodeObjects,
                           KFMElectrostaticElementContainerBase<KFMELECTROSTATICS_DIM, KFMELECTROSTATICS_BASIS>>
        elementContainerNullifier;
    KFMNodeObjectNullifier<KFMElectrostaticNodeObjects, KFMCubicSpaceTreeProperties<KFMELECTROSTATICS_DIM>>
        treePropertyNullifier;

    KFMRecursiveActor<KFMElectrostaticNode> recursiveActor;

    for (unsigned int i = 0; i < fNonTargetNodeCollection.size(); i++) {
        //have to null out the pointer to the element container
        //so the node's do not try to delete it when they are destroyed
        recursiveActor.SetOperationalActor(&elementContainerNullifier);
        recursiveActor.ApplyAction(fNonTargetNodeCollection[i]);

        recursiveActor.SetOperationalActor(&treePropertyNullifier);
        recursiveActor.ApplyAction(fNonTargetNodeCollection[i]);

        //now delete all of the children of this node_collection
        fNonTargetNodeCollection[i]->DeleteChildren();

        //reset the pointer to the element container for the top level nodes
        KFMObjectRetriever<KFMElectrostaticNodeObjects,
                           KFMElectrostaticElementContainerBase<KFMELECTROSTATICS_DIM, KFMELECTROSTATICS_BASIS>>::
            SetNodeObject(fContainer, fNonTargetNodeCollection[i]);

        //reset the pointer to the cubic space tree properties for the top level nodes
        KFMCubicSpaceTreeProperties<KFMELECTROSTATICS_DIM>* tree_prop = NULL;
        tree_prop =
            KFMObjectRetriever<KFMElectrostaticNodeObjects,
                               KFMCubicSpaceTreeProperties<KFMELECTROSTATICS_DIM>>::GetNodeObject(fTree->GetRootNode());
        KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCubicSpaceTreeProperties<KFMELECTROSTATICS_DIM>>::
            SetNodeObject(tree_prop, fNonTargetNodeCollection[i]);
    }
}

////////////////////////////////////////////////////////////////////////////////


void KFMElectrostaticTreeBuilder_MPI::CollectDirectCallIdentitiesForPrimaryNodes()
{
    fTree->RestrictActionBehavior(false);

    //sort id sets
    KFMElectrostaticIdentitySetSorter IDSorter;
    fTree->ApplyRecursiveAction(&IDSorter);

    //the primacy flag inspector
    KFMNodeFlagValueInspector<KFMElectrostaticNodeObjects, KFMELECTROSTATICS_FLAGS> primary_flag_condition;
    primary_flag_condition.SetFlagIndex(0);
    primary_flag_condition.SetFlagValue(1);

    //we assume that the primary nodes have already been determined
    //create a conditional actor, which depends on node primacy
    //to construct the direct call lists
    KFMConditionalActor<KFMElectrostaticNode> conditional_actor;
    conditional_actor.SetInspectingActor(&primary_flag_condition);

    //create the lists of the external identity sets
    //(reduced memory alternative to explicitly creating the external id sets as above)
    KFMElectrostaticIdentitySetListCreator IDListCreator;
    IDListCreator.SetZeroMaskSize(fZeroMaskSize);
    conditional_actor.SetOperationalActor(&IDListCreator);
    //apply action
    fTree->ApplyCorecursiveAction(&conditional_actor);

    //now we create the lists of the collocation points
    //these are associated with the centroid of each electrostatic element
    //first we form a list of the centroids
    std::vector<const KFMPoint<KFMELECTROSTATICS_DIM>*> centroids;
    std::vector<unsigned int> centroid_ids;
    unsigned int n_elements = fContainer->GetNElements();
    for (unsigned int i = 0; i < n_elements; i++) {
        if (fTargetVolume.PointIsInside(*(fContainer->GetCentroid(i)))) {
            centroid_ids.push_back(i);
            centroids.push_back(fContainer->GetCentroid(i));
        }
    }

    KFMElectrostaticCollocationPointIdentitySetCreator cpid_creator;
    cpid_creator.SetTree(fTree);
    cpid_creator.SortCollocationPoints(&centroids, &centroid_ids);
}


}  // namespace KEMField
