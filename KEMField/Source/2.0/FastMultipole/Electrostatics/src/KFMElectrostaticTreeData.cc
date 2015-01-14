#include "KFMElectrostaticTreeData.hh"

namespace KEMField
{

void
KFMElectrostaticTreeData::GetFlattenedTree(std::vector<KFMNodeData>* node_data) const
{
    *node_data = fFlattenedTree;
}

void
KFMElectrostaticTreeData::SetFlattenedTree(const std::vector<KFMNodeData>* node_data)
{
    fFlattenedTree = *node_data;
}

void
KFMElectrostaticTreeData::GetLocalCoefficientNodeIDs(std::vector<int>* node_ids) const
{
    *node_ids = fLocalCoefficientsNodeIDs;
}

void
KFMElectrostaticTreeData::SetLocalCoefficientNodeIDs(const std::vector<int>* node_ids)
{
    fLocalCoefficientsNodeIDs = *node_ids;
}

void
KFMElectrostaticTreeData::GetLocalCoefficients(std::vector< KFMElectrostaticLocalCoefficientSet* >* local_coeff) const
{
    *local_coeff = fLocalCoefficients;
}

void
KFMElectrostaticTreeData::SetLocalCoefficients(const std::vector< KFMElectrostaticLocalCoefficientSet* >* local_coeff)
{
    fLocalCoefficients = *local_coeff;
}

void
KFMElectrostaticTreeData::GetIdentitySetNodeIDs(std::vector<int>* node_ids) const
{
    *node_ids = fIdentitySetNodeIDs;
}

void
KFMElectrostaticTreeData::SetIdentitySetNodeIDs(const std::vector<int>* node_ids)
{
    fIdentitySetNodeIDs = *node_ids;
}

void
KFMElectrostaticTreeData::GetIdentitySets(std::vector< KFMIdentitySet* >* id_sets) const
{
    *id_sets = fIdentitySets;
}

void
KFMElectrostaticTreeData::SetIdentitySets(const std::vector< KFMIdentitySet* >* id_sets)
{
    fIdentitySets = *id_sets;
}

void
KFMElectrostaticTreeData::GetCubeNodeIDs(std::vector<int>* node_ids) const
{
    *node_ids = fCubeNodeIDs;
}

void
KFMElectrostaticTreeData::SetCubeNodeIDs(const std::vector<int>* node_ids)
{
    fCubeNodeIDs = *node_ids;
}


void
KFMElectrostaticTreeData::GetCubes(std::vector< KFMCube<3>* >* cubes) const
{
    *cubes = fCubes;
}

void
KFMElectrostaticTreeData::SetCubes(const std::vector< KFMCube<3>* >* cubes)
{
    fCubes = *cubes;
}

void
KFMElectrostaticTreeData::DefineOutputNode(KSAOutputNode* node) const
{
    if(node != NULL)
    {
        node->AddChild(new KSAAssociatedValuePODOutputNode< KFMElectrostaticTreeData, std::string, &KFMElectrostaticTreeData::GetTreeID >( std::string("tree_id"), this) );
        node->AddChild(new KSAAssociatedValuePODOutputNode< KFMElectrostaticTreeData, unsigned int, &KFMElectrostaticTreeData::GetNumberOfTreeNodes >( std::string("n_tree_nodes"), this) );
        node->AddChild(new KSAAssociatedValuePODOutputNode< KFMElectrostaticTreeData, unsigned int, &KFMElectrostaticTreeData::GetDivisions >( std::string("divisions"), this) );
        node->AddChild(new KSAAssociatedValuePODOutputNode< KFMElectrostaticTreeData, unsigned int, &KFMElectrostaticTreeData::GetDegree >( std::string("degree"), this) );
        node->AddChild(new KSAAssociatedValuePODOutputNode< KFMElectrostaticTreeData, unsigned int, &KFMElectrostaticTreeData::GetZeroMaskSize >( std::string("zero_mask_size"), this) );
        node->AddChild(new KSAAssociatedValuePODOutputNode< KFMElectrostaticTreeData, unsigned int, &KFMElectrostaticTreeData::GetMaximumTreeDepth >( std::string("maximum_tree_depth"), this) );
        node->AddChild(new KSAAssociatedValuePODOutputNode< KFMElectrostaticTreeData, unsigned int, &KFMElectrostaticTreeData::GetMaxDirectCalls >( std::string("max_direct_calls"), this) );


        node->AddChild( new KSAAssociatedPassedPointerPODOutputNode< KFMElectrostaticTreeData, std::vector< int >, &KFMElectrostaticTreeData::GetLocalCoefficientNodeIDs >(std::string("LocalCoefficientNodeIDVector"), this) );
        node->AddChild( new KSAAssociatedPassedPointerPODOutputNode< KFMElectrostaticTreeData, std::vector< int >, &KFMElectrostaticTreeData::GetIdentitySetNodeIDs >(std::string("IdentitySetNodeIDVector"), this) );
        node->AddChild( new KSAAssociatedPassedPointerPODOutputNode< KFMElectrostaticTreeData, std::vector< int >, &KFMElectrostaticTreeData::GetCubeNodeIDs >(std::string("CubeNodeIDVector"), this) );


        node->AddChild(new KSAObjectOutputNode< std::vector< KFMNodeData > >( std::string("NodeDataVector"), &fFlattenedTree) );
        node->AddChild(new KSAObjectOutputNode< std::vector< KFMIdentitySet* > >( std::string("IdentitySetVector"), &fIdentitySets) );
        node->AddChild(new KSAObjectOutputNode< std::vector< KFMCube<3>* > >( std::string("CubeVector"), &fCubes) );
        node->AddChild(new KSAObjectOutputNode< std::vector< KFMElectrostaticLocalCoefficientSet* > >( std::string("LocalCoefficientVector"), &fLocalCoefficients) );
    }
}

void
KFMElectrostaticTreeData::DefineInputNode(KSAInputNode* node)
{
    if(node != NULL)
    {
        node->AddChild(new KSAAssociatedReferencePODInputNode< KFMElectrostaticTreeData, std::string, &KFMElectrostaticTreeData::SetTreeID >( std::string("tree_id"), this) );
        node->AddChild(new KSAAssociatedReferencePODInputNode< KFMElectrostaticTreeData, unsigned int, &KFMElectrostaticTreeData::SetNumberOfTreeNodes >( std::string("n_tree_nodes"), this) );
        node->AddChild(new KSAAssociatedReferencePODInputNode< KFMElectrostaticTreeData, unsigned int, &KFMElectrostaticTreeData::SetDivisions >( std::string("divisions"), this) );
        node->AddChild(new KSAAssociatedReferencePODInputNode< KFMElectrostaticTreeData, unsigned int, &KFMElectrostaticTreeData::SetDegree >( std::string("degree"), this) );
        node->AddChild(new KSAAssociatedReferencePODInputNode< KFMElectrostaticTreeData, unsigned int, &KFMElectrostaticTreeData::SetZeroMaskSize >( std::string("zero_mask_size"), this) );
        node->AddChild(new KSAAssociatedReferencePODInputNode< KFMElectrostaticTreeData, unsigned int, &KFMElectrostaticTreeData::SetMaximumTreeDepth >( std::string("maximum_tree_depth"), this) );
        node->AddChild(new KSAAssociatedReferencePODInputNode< KFMElectrostaticTreeData, unsigned int, &KFMElectrostaticTreeData::SetMaxDirectCalls >( std::string("max_direct_calls"), this) );

        node->AddChild( new KSAAssociatedPointerPODInputNode<KFMElectrostaticTreeData, std::vector< int >, &KFMElectrostaticTreeData::SetLocalCoefficientNodeIDs >(std::string("LocalCoefficientNodeIDVector"), this) );
        node->AddChild( new KSAAssociatedPointerPODInputNode<KFMElectrostaticTreeData, std::vector< int >, &KFMElectrostaticTreeData::SetIdentitySetNodeIDs >(std::string("IdentitySetNodeIDVector"), this) );
        node->AddChild( new KSAAssociatedPointerPODInputNode<KFMElectrostaticTreeData, std::vector< int >, &KFMElectrostaticTreeData::SetCubeNodeIDs >(std::string("CubeNodeIDVector"), this) );

        node->AddChild(new KSAObjectInputNode< std::vector< KFMNodeData > >( std::string("NodeDataVector"), &fFlattenedTree) );
        node->AddChild(new KSAObjectInputNode< std::vector< KFMIdentitySet* > >( std::string("IdentitySetVector"), &fIdentitySets) );
        node->AddChild(new KSAObjectInputNode< std::vector< KFMCube<3>* > >( std::string("CubeVector"), &fCubes) );
        node->AddChild(new KSAObjectInputNode< std::vector< KFMElectrostaticLocalCoefficientSet* > >( std::string("LocalCoefficientVector"), &fLocalCoefficients) );
    }
}


}
