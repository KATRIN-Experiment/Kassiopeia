#ifndef KFMElectrostaticTreeConstructor_HH__
#define KFMElectrostaticTreeConstructor_HH__

//fm electrostatics
#include "KFMElectrostaticFieldMapper_SingleThread.hh"
#include "KFMElectrostaticParameters.hh"
#include "KFMElectrostaticTree.hh"
#include "KFMElectrostaticTreeBuilder.hh"
#include "KFMObjectCollector.hh"

//fm interface extraction
#include "KFMElectrostaticElement.hh"
#include "KFMElectrostaticElementContainer.hh"
#include "KFMElectrostaticElementContainerBase.hh"
#include "KFMElectrostaticSurfaceConverter.hh"

//IO
#include "KFMElectrostaticTreeData.hh"
#include "KFMNodeData.hh"
#include "KFMTreeStructureExtractor.hh"

//surfaces
#include "KSortedSurfaceContainer.hh"
#include "KSurfaceContainer.hh"

namespace KEMField
{

/*
*
*@file KFMElectrostaticTreeConstructor.hh
*@class KFMElectrostaticTreeConstructor
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Mar 14 15:03:57 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ParallelTrait = KFMElectrostaticFieldMapper_SingleThread> class KFMElectrostaticTreeConstructor
{
  public:
    KFMElectrostaticTreeConstructor(){};
    virtual ~KFMElectrostaticTreeConstructor(){};


    void ConstructTree(const KSurfaceContainer& surface_container, KFMElectrostaticTree& tree) const
    {
        KFMElectrostaticParameters params = tree.GetParameters();

        if (params.verbosity > 0) {
            kfmout
                << "KFMElectrostaticTreeConstructor::ConstructTree: Constructing fast multipole region tree from surface container."
                << kfmendl;
        }
        //now we have a surface container with a bunch of electrode discretizations
        //we just want to convert these into point clouds, and then bounding balls
        //extract the information we want

        KFMElectrostaticElementContainerBase<3, 1>* elementContainer;
        elementContainer = new KFMElectrostaticElementContainer<3, 1>();
        KFMElectrostaticSurfaceConverter surfaceConverter;
        surfaceConverter.SetSurfaceContainer(&surface_container);
        surfaceConverter.SetElectrostaticElementContainer(elementContainer);
        surfaceConverter.Extract();

        //create the tree builder
        KFMElectrostaticTreeBuilder treeBuilder;
        treeBuilder.SetElectrostaticElementContainer(elementContainer);
        treeBuilder.SetTree(&tree);

        //now we construct the tree's structure
        treeBuilder.ConstructRootNode();
        treeBuilder.PerformSpatialSubdivision();
        treeBuilder.FlagNonZeroMultipoleNodes();
        treeBuilder.PerformAdjacencySubdivision();
        treeBuilder.FlagPrimaryNodes();
        treeBuilder.CollectDirectCallIdentities();

        //the parallel trait is responsible for computing
        //local coefficient field map everywhere it is needed (leaf nodes)
        //and performs all clean up needed afterwards
        ParallelTrait trait;
        trait.SetElectrostaticElementContainer(elementContainer);
        trait.SetTree(&tree);
        trait.Initialize();
        trait.MapField();

        //clean up
        delete elementContainer;
    }


    ////////////////////////////////////////////////////////////////////////////////

    void ConstructTree(const KSortedSurfaceContainer& surface_container, KFMElectrostaticTree& tree) const
    {
        KFMElectrostaticParameters params = tree.GetParameters();

        if (params.verbosity > 0) {
            kfmout
                << "KFMElectrostaticTreeConstructor::ConstructTree: Constructing fast multipole region tree from sorted surface container."
                << kfmendl;
        }

        //now we have a surface container with a bunch of electrode discretizations
        //we just want to convert these into point clouds, and then bounding balls
        //extract the information we want
        KFMElectrostaticElementContainerBase<3, 1>* elementContainer;
        elementContainer = new KFMElectrostaticElementContainer<3, 1>();

        KFMElectrostaticSurfaceConverter surfaceConverter;
        surfaceConverter.SetSortedSurfaceContainer(&surface_container);
        surfaceConverter.SetElectrostaticElementContainer(elementContainer);
        surfaceConverter.Extract();

        //create the tree builder
        KFMElectrostaticTreeBuilder treeBuilder;
        treeBuilder.SetElectrostaticElementContainer(elementContainer);
        treeBuilder.SetTree(&tree);

        //now we construct the tree's structure
        treeBuilder.ConstructRootNode();
        treeBuilder.PerformSpatialSubdivision();
        treeBuilder.FlagNonZeroMultipoleNodes();
        treeBuilder.PerformAdjacencySubdivision();
        treeBuilder.FlagPrimaryNodes();
        treeBuilder.CollectDirectCallIdentities();

        //the parallel trait is responsible for computing
        //local coefficient field map everywhere it is needed (leaf nodes)
        //and performs all clean up needed afterwards
        ParallelTrait trait;
        trait.SetElectrostaticElementContainer(elementContainer);
        trait.SetTree(&tree);
        trait.Initialize();
        trait.MapField();

        delete elementContainer;
    }

    ////////////////////////////////////////////////////////////////////////////////

    void ConstructTree(KFMElectrostaticTreeData& data, KFMElectrostaticTree& tree) const
    {
        //now we have to re-build the tree from the objects now in memory
        //first retrieve the parameters
        unsigned int top_level_divisions = data.GetTopLevelDivisions();
        unsigned int divisions = data.GetDivisions();
        unsigned int top_div[3];
        top_div[0] = top_level_divisions;
        top_div[1] = top_level_divisions;
        top_div[2] = top_level_divisions;
        unsigned int div[3];
        div[0] = divisions;
        div[1] = divisions;
        div[2] = divisions;
        unsigned int degree = data.GetDegree();
        unsigned int zeromask = data.GetZeroMaskSize();
        unsigned int max_tree_depth = data.GetMaximumTreeDepth();
        double insertion_ratio = data.GetInsertionRatio();
        unsigned int n_nodes = data.GetNumberOfTreeNodes();

        KFMElectrostaticParameters params;
        params.top_level_divisions = top_level_divisions;
        params.divisions = divisions;
        params.degree = degree;
        params.zeromask = zeromask;
        params.maximum_tree_depth = max_tree_depth;
        params.insertion_ratio = insertion_ratio;

        tree.SetParameters(params);

        //access and modify the tree's properties
        KFMCubicSpaceTreeProperties<3>* tree_prop = tree.GetTreeProperties();

        tree_prop->SetTreeID(data.GetTreeID());
        tree_prop->SetMaxTreeDepth(max_tree_depth);
        tree_prop->SetCubicNeighborOrder(zeromask);
        tree_prop->SetDimensions(div);
        tree_prop->SetTopLevelDimensions(top_div);

        //now we need to create a vector of N empty nodes, and enumerate them
        std::vector<KFMElectrostaticNode*> tree_nodes;
        tree_nodes.resize(n_nodes, nullptr);

        for (unsigned int i = 0; i < n_nodes; i++) {
            tree_nodes[i] = new KFMElectrostaticNode();
            tree_nodes[i]->SetID(i);

            //attach the tree properties to these nodes
            KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCubicSpaceTreeProperties<3>>::SetNodeObject(
                tree_prop,
                tree_nodes[i]);
        }

        //now we need to re-link the tree, by connecting child nodes to their parents
        std::vector<KFMNodeData> tree_structure_data;
        data.GetFlattenedTree(&tree_structure_data);

        for (unsigned int i = 0; i < tree_structure_data.size(); i++) {
            KFMElectrostaticNode* current_node = tree_nodes[tree_structure_data[i].GetID()];

            //link the children of this current node
            std::vector<unsigned int> child_ids;
            tree_structure_data[i].GetChildIDs(&child_ids);
            for (unsigned int j = 0; j < child_ids.size(); j++) {
                current_node->AddChild(tree_nodes[child_ids[j]]);
            }
        }

        //now we need to re-attach objects to the node which owns them
        std::vector<int> local_coeff_node_ids;
        std::vector<KFMElectrostaticLocalCoefficientSet*> local_coeff;
        data.GetLocalCoefficientNodeIDs(&local_coeff_node_ids);
        data.GetLocalCoefficients(&local_coeff);

        for (unsigned int i = 0; i < local_coeff_node_ids.size(); i++) {
            KFMElectrostaticNode* node = tree_nodes[local_coeff_node_ids[i]];
            KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMElectrostaticLocalCoefficientSet>::SetNodeObject(
                local_coeff[i],
                node);
        }

        std::vector<int> id_set_node_ids;
        std::vector<KFMIdentitySet*> id_sets;
        data.GetIdentitySetNodeIDs(&id_set_node_ids);
        data.GetIdentitySets(&id_sets);

        for (unsigned int i = 0; i < id_set_node_ids.size(); i++) {
            KFMElectrostaticNode* node = tree_nodes[id_set_node_ids[i]];
            KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMIdentitySet>::SetNodeObject(id_sets[i], node);
        }


        std::vector<int> cube_node_ids;
        std::vector<KFMCube<3>*> cubes;
        data.GetCubeNodeIDs(&cube_node_ids);
        data.GetCubes(&cubes);

        for (unsigned int i = 0; i < cube_node_ids.size(); i++) {
            KFMElectrostaticNode* node = tree_nodes[cube_node_ids[i]];
            KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCube<3>>::SetNodeObject(cubes[i], node);
        }

        //now replace the tree's root node with the new one
        tree.ReplaceRootNode(tree_nodes[0]);

        //now we reconstruct the direct call lists
        KFMElectrostaticIdentitySetListCreator IDListCreator;
        IDListCreator.SetZeroMaskSize(params.zeromask);
        tree.ApplyCorecursiveAction(&IDListCreator);
    }

    //////////////////////////////////////////////////////////////////////////////////


    void SaveTree(KFMElectrostaticTree& tree, KFMElectrostaticTreeData& data) const
    {
        //extract to temporary holder needed to stream the tree data
        data.SetTreeID(tree.GetTreeProperties()->GetTreeID());

        //get the tree parameters to store them
        KFMElectrostaticParameters params = tree.GetParameters();

        data.SetTopLevelDivisions(params.top_level_divisions);
        data.SetDivisions(params.divisions);
        data.SetDegree(params.degree);
        data.SetZeroMaskSize(params.zeromask);
        data.SetMaximumTreeDepth(params.maximum_tree_depth);
        data.SetInsertionRatio(params.insertion_ratio);

        //first we need to flatten the tree structure before we can stream it
        KFMTreeStructureExtractor<KFMElectrostaticNodeObjects> flattener;
        tree.ApplyRecursiveAction(&flattener);
        data.SetNumberOfTreeNodes(flattener.GetNumberOfNodes());
        data.SetFlattenedTree(flattener.GetFlattenedTree());

        //now we are going to extract the pointers to the objects we want to save
        KFMObjectCollector<KFMElectrostaticNodeObjects, KFMElectrostaticLocalCoefficientSet> coeff_collector;
        tree.ApplyCorecursiveAction(&coeff_collector);

        KFMObjectCollector<KFMElectrostaticNodeObjects, KFMIdentitySet> id_set_collector;
        tree.ApplyCorecursiveAction(&id_set_collector);

        KFMObjectCollector<KFMElectrostaticNodeObjects, KFMCube<3>> cube_collector;
        tree.ApplyCorecursiveAction(&cube_collector);

        //copy pointers to data container
        data.SetLocalCoefficientNodeIDs(coeff_collector.GetCollectedObjectAssociatedNodeIDs());
        data.SetLocalCoefficients(coeff_collector.GetCollectedObjects());

        data.SetIdentitySetNodeIDs(id_set_collector.GetCollectedObjectAssociatedNodeIDs());
        data.SetIdentitySets(id_set_collector.GetCollectedObjects());

        data.SetCubeNodeIDs(cube_collector.GetCollectedObjectAssociatedNodeIDs());
        data.SetCubes(cube_collector.GetCollectedObjects());
    }


  protected:
};


}  // namespace KEMField

#endif /* KFMElectrostaticTreeConstructor_H__ */
