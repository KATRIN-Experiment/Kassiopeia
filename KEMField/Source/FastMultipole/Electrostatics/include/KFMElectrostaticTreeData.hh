#ifndef KFMElectrostaticTreeData_HH__
#define KFMElectrostaticTreeData_HH__

#include "KFMElectrostaticNode.hh"
#include "KFMElectrostaticTree.hh"
#include "KFMNode.hh"
#include "KFMNodeData.hh"
#include "KSAStructuredASCIIHeaders.hh"

#include <vector>

namespace KEMField
{


/*
*
*@file KFMElectrostaticTreeData.hh
*@class KFMElectrostaticTreeData
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Apr  2 22:45:19 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMElectrostaticTreeData : public KSAInputOutputObject
{
  public:
    KFMElectrostaticTreeData()
    {
        fTopLevelDivisions = 0;
        fDivisions = 0;
        fDegree = 0;
        fZeroMaskSize = 0;
        fMaxTreeDepth = 0;
        fInsertionRatio = 0;
        fNNodes = 0;
    };
    ~KFMElectrostaticTreeData() override{};

    std::string GetTreeID() const
    {
        return fTreeID;
    };
    void SetTreeID(const std::string& id)
    {
        fTreeID = id;
    };

    void GetFlattenedTree(std::vector<KFMNodeData>* node_data) const;
    void SetFlattenedTree(const std::vector<KFMNodeData>* node_data);
    const std::vector<KFMNodeData>* GetFlattenedTreePointer() const
    {
        return &fFlattenedTree;
    };
    std::vector<KFMNodeData>* GetFlattenedTreePointer()
    {
        return &fFlattenedTree;
    };


    void GetLocalCoefficientNodeIDs(std::vector<int>* node_ids) const;
    void SetLocalCoefficientNodeIDs(const std::vector<int>* node_ids);
    const std::vector<int>* GetLocalCoefficientNodeIDPointer() const
    {
        return &fLocalCoefficientsNodeIDs;
    };
    std::vector<int>* GetLocalCoefficientNodeIDPointer()
    {
        return &fLocalCoefficientsNodeIDs;
    };


    void GetLocalCoefficients(std::vector<KFMElectrostaticLocalCoefficientSet*>* local_coeff) const;
    void SetLocalCoefficients(const std::vector<KFMElectrostaticLocalCoefficientSet*>* local_coeff);
    const std::vector<KFMElectrostaticLocalCoefficientSet*>* GetLocalCoefficientPointer() const
    {
        return &fLocalCoefficients;
    };
    std::vector<KFMElectrostaticLocalCoefficientSet*>* GetLocalCoefficientPointer()
    {
        return &fLocalCoefficients;
    };


    void GetIdentitySetNodeIDs(std::vector<int>* node_ids) const;
    void SetIdentitySetNodeIDs(const std::vector<int>* node_ids);
    const std::vector<int>* GetIdentitySetNodeIDPointer() const
    {
        return &fIdentitySetNodeIDs;
    };
    std::vector<int>* GetIdentitySetNodeIDPointer()
    {
        return &fIdentitySetNodeIDs;
    };


    void GetIdentitySets(std::vector<KFMIdentitySet*>* id_sets) const;
    void SetIdentitySets(const std::vector<KFMIdentitySet*>* id_sets);
    const std::vector<KFMIdentitySet*>* GetIdentitySetPointer() const
    {
        return &fIdentitySets;
    };
    std::vector<KFMIdentitySet*>* GetIdentitySetPointer()
    {
        return &fIdentitySets;
    };

    void GetCubeNodeIDs(std::vector<int>* node_ids) const;
    void SetCubeNodeIDs(const std::vector<int>* node_ids);
    const std::vector<int>* GetCubeNodeIDPointer() const
    {
        return &fCubeNodeIDs;
    };
    std::vector<int>* GetCubeNodeIDPointer()
    {
        return &fCubeNodeIDs;
    };

    void GetCubes(std::vector<KFMCube<3>*>* cubes) const;
    void SetCubes(const std::vector<KFMCube<3>*>* cubes);
    const std::vector<KFMCube<3>*>* GetCubePointer() const
    {
        return &fCubes;
    };
    std::vector<KFMCube<3>*>* GetCubePointer()
    {
        return &fCubes;
    };


    unsigned int GetNumberOfTreeNodes() const
    {
        return fNNodes;
    };
    void SetNumberOfTreeNodes(const unsigned int& nnodes)
    {
        fNNodes = nnodes;
    };

    //Parameters used when building the tree, that we will need when reconstructing it
    unsigned int GetTopLevelDivisions() const
    {
        return fTopLevelDivisions;
    };
    void SetTopLevelDivisions(const unsigned int& div)
    {
        fTopLevelDivisions = div;
    };

    unsigned int GetDivisions() const
    {
        return fDivisions;
    };
    void SetDivisions(const unsigned int& div)
    {
        fDivisions = div;
    };

    unsigned int GetDegree() const
    {
        return fDegree;
    };
    void SetDegree(const unsigned int& deg)
    {
        fDegree = deg;
    };

    unsigned int GetZeroMaskSize() const
    {
        return fZeroMaskSize;
    };
    void SetZeroMaskSize(const unsigned int& zm)
    {
        fZeroMaskSize = zm;
    };

    unsigned int GetMaximumTreeDepth() const
    {
        return fMaxTreeDepth;
    };
    void SetMaximumTreeDepth(const unsigned int& max_depth)
    {
        fMaxTreeDepth = max_depth;
    };

    double GetInsertionRatio() const
    {
        return fInsertionRatio;
    };
    void SetInsertionRatio(const double& insertion_ratio)
    {
        fInsertionRatio = insertion_ratio;
    };


    //IO
    virtual std::string ClassName() const
    {
        return std::string("KFMElectrostaticTreeData");
    };
    static std::string Name()
    {
        return std::string("KFMElectrostaticTreeData");
    };
    void DefineOutputNode(KSAOutputNode* node) const override;
    void DefineInputNode(KSAInputNode* node) override;


  private:
    //parameters
    unsigned int fTopLevelDivisions;
    unsigned int fDivisions;
    unsigned int fDegree;
    unsigned int fZeroMaskSize;
    unsigned int fMaxTreeDepth;
    double fInsertionRatio;

    //storage for the tree structure
    std::string fTreeID;
    unsigned int fNNodes;
    std::vector<KFMNodeData> fFlattenedTree;

    std::vector<int> fLocalCoefficientsNodeIDs;
    std::vector<KFMElectrostaticLocalCoefficientSet*> fLocalCoefficients;

    std::vector<int> fIdentitySetNodeIDs;
    std::vector<KFMIdentitySet*> fIdentitySets;

    std::vector<int> fCubeNodeIDs;
    std::vector<KFMCube<3>*> fCubes;
};


template<typename Stream> Stream& operator>>(Stream& s, KFMElectrostaticTreeData& aData)
{
    s.PreStreamInAction(aData);

    unsigned int tdiv;
    s >> tdiv;
    aData.SetTopLevelDivisions(tdiv);

    unsigned int div;
    s >> div;
    aData.SetDivisions(div);

    unsigned int deg;
    s >> deg;
    aData.SetDegree(deg);

    unsigned int zmask;
    s >> zmask;
    aData.SetZeroMaskSize(zmask);

    unsigned int tree_depth;
    s >> tree_depth;
    aData.SetMaximumTreeDepth(tree_depth);

    double insertion_ratio;
    s >> insertion_ratio;
    aData.SetInsertionRatio(insertion_ratio);

    std::string tree_id;
    s >> tree_id;
    aData.SetTreeID(tree_id);

    unsigned int n_tree_nodes;
    s >> n_tree_nodes;
    aData.SetNumberOfTreeNodes(n_tree_nodes);

    std::vector<KFMNodeData>* flattened_tree = aData.GetFlattenedTreePointer();
    unsigned int n_flattened_nodes;
    s >> n_flattened_nodes;
    flattened_tree->resize(0);
    flattened_tree->reserve(n_flattened_nodes);
    for (unsigned int i = 0; i < n_flattened_nodes; i++) {
        KFMNodeData node_data;
        s >> node_data;
        flattened_tree->push_back(node_data);
    }

    std::vector<int>* local_coeff_node_ids = aData.GetLocalCoefficientNodeIDPointer();
    unsigned int local_coeff_node_ids_size;
    s >> local_coeff_node_ids_size;
    local_coeff_node_ids->resize(0);
    local_coeff_node_ids->reserve(local_coeff_node_ids_size);
    for (unsigned int i = 0; i < local_coeff_node_ids_size; i++) {
        unsigned int id;
        s >> id;
        local_coeff_node_ids->push_back(id);
    }

    std::vector<KFMElectrostaticLocalCoefficientSet*>* local_coeff = aData.GetLocalCoefficientPointer();
    unsigned int local_coeff_size;
    s >> local_coeff_size;
    local_coeff->resize(0);
    local_coeff->reserve(local_coeff_size);
    for (unsigned int i = 0; i < local_coeff_size; i++) {
        auto* set = new KFMElectrostaticLocalCoefficientSet();
        s >> *set;
        local_coeff->push_back(set);
    }


    std::vector<int>* id_set_node_ids = aData.GetIdentitySetNodeIDPointer();
    unsigned int id_set_node_ids_size;
    s >> id_set_node_ids_size;
    id_set_node_ids->resize(0);
    id_set_node_ids->reserve(id_set_node_ids_size);
    for (unsigned int i = 0; i < id_set_node_ids_size; i++) {
        unsigned int id;
        s >> id;
        id_set_node_ids->push_back(id);
    }

    std::vector<KFMIdentitySet*>* id_set = aData.GetIdentitySetPointer();
    unsigned int id_set_size;
    s >> id_set_size;
    id_set->resize(0);
    id_set->reserve(id_set_size);
    for (unsigned int i = 0; i < id_set_size; i++) {
        auto* set = new KFMIdentitySet();
        s >> *set;
        id_set->push_back(set);
    }

    std::vector<int>* cube_node_ids = aData.GetCubeNodeIDPointer();
    unsigned int cube_node_ids_size;
    s >> cube_node_ids_size;
    cube_node_ids->resize(0);
    cube_node_ids->reserve(cube_node_ids_size);
    for (unsigned int i = 0; i < cube_node_ids_size; i++) {
        unsigned int id;
        s >> id;
        cube_node_ids->push_back(id);
    }

    std::vector<KFMCube<3>*>* cubes = aData.GetCubePointer();
    unsigned int cube_size;
    s >> cube_size;
    cubes->resize(0);
    cubes->reserve(cube_size);
    for (unsigned int i = 0; i < cube_size; i++) {
        auto* cube = new KFMCube<3>();
        s >> *cube;
        cubes->push_back(cube);
    }

    s.PostStreamInAction(aData);
    return s;
}

template<typename Stream> Stream& operator<<(Stream& s, const KFMElectrostaticTreeData& aData)
{
    s.PreStreamOutAction(aData);

    s << aData.GetTopLevelDivisions();
    s << aData.GetDivisions();
    s << aData.GetDegree();
    s << aData.GetZeroMaskSize();
    s << aData.GetMaximumTreeDepth();
    s << aData.GetInsertionRatio();
    s << aData.GetTreeID();

    unsigned int n_tree_nodes = aData.GetNumberOfTreeNodes();
    s << n_tree_nodes;

    const std::vector<KFMNodeData>* flattened_tree = aData.GetFlattenedTreePointer();
    unsigned int n_flattened_nodes = flattened_tree->size();
    s << n_flattened_nodes;
    for (unsigned int i = 0; i < n_flattened_nodes; i++) {
        s << (*flattened_tree)[i];
    }

    const std::vector<int>* local_coeff_node_ids = aData.GetLocalCoefficientNodeIDPointer();
    unsigned int local_coeff_node_ids_size = local_coeff_node_ids->size();
    s << local_coeff_node_ids_size;
    for (unsigned int i = 0; i < local_coeff_node_ids_size; i++) {
        s << (*local_coeff_node_ids)[i];
    }

    const std::vector<KFMElectrostaticLocalCoefficientSet*>* local_coeff = aData.GetLocalCoefficientPointer();
    unsigned int local_coeff_size = local_coeff->size();
    s << local_coeff_size;
    for (unsigned int i = 0; i < local_coeff_size; i++) {
        s << *((*local_coeff)[i]);
    }

    const std::vector<int>* id_set_node_ids = aData.GetIdentitySetNodeIDPointer();
    unsigned int id_set_node_ids_size = id_set_node_ids->size();
    s << id_set_node_ids_size;
    for (unsigned int i = 0; i < id_set_node_ids_size; i++) {
        s << (*id_set_node_ids)[i];
    }


    const std::vector<KFMIdentitySet*>* id_set = aData.GetIdentitySetPointer();
    unsigned int id_set_size = id_set->size();
    s << id_set_size;
    for (unsigned int i = 0; i < id_set_size; i++) {
        s << *((*id_set)[i]);
    }

    const std::vector<int>* cube_node_ids = aData.GetCubeNodeIDPointer();
    unsigned int cube_node_ids_size = cube_node_ids->size();
    s << cube_node_ids_size;
    for (unsigned int i = 0; i < cube_node_ids_size; i++) {
        s << (*cube_node_ids)[i];
    }


    const std::vector<KFMCube<3>*>* cubes = aData.GetCubePointer();
    unsigned int cube_size = cubes->size();
    s << cube_size;
    for (unsigned int i = 0; i < cube_size; i++) {
        s << *((*cubes)[i]);
    }


    s.PostStreamOutAction(aData);

    return s;
}

DefineKSAClassName(KFMElectrostaticTreeData)

}  // namespace KEMField

#endif /* KFMElectrostaticTreeData_H__ */
