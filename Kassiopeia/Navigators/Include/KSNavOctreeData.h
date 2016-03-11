#ifndef KSNavOctreeData_HH__
#define KSNavOctreeData_HH__

#include <vector>

#include "KGNode.hh"
#include "KGNodeData.hh"

#include "KGNavigableMeshElement.hh"
#include "KGMeshNavigationNode.hh"
#include "KGNavigableMeshTree.hh"

namespace Kassiopeia
{

/*
*
*@file KSNavOctreeData.hh
*@class KSNavOctreeData
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Apr  2 22:45:19 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KSNavOctreeData
{
    public:

        KSNavOctreeData():
            fMaxDepth(0),
            fSpecifyMaxDepth(0),
            fSpatialResolution(0),
            fSpecifyResolution(0),
            fNAllowedElements(0),
            fSpecifyAllowedElements(0),
            fTreeID(""),
            fNNodes(0)
        {
            fFlattenedTree.clear();

            fIdentitySetNodeIDs.clear();
            fIdentitySets.clear();

            fCubeNodeIDs.clear();
            fCubes.clear();
        }

        virtual ~KSNavOctreeData(){};

        static std::string Name() { return std::string("octree_data");};

        std::string GetTreeID() const {return fTreeID;};
        void SetTreeID(const std::string& id){fTreeID = id;};

        void SetNumberOfTreeNodes(unsigned int n_nodes){fNNodes = n_nodes;};
        unsigned int GetNumberOfTreeNodes() const {return fNNodes;};

        void GetFlattenedTree(std::vector<KGNodeData>* node_data) const {*node_data = fFlattenedTree;};
        void SetFlattenedTree(const std::vector<KGNodeData>* node_data) {fFlattenedTree = *node_data;};
        const std::vector<KGNodeData>* GetFlattenedTreePointer() const {return &fFlattenedTree;};
        std::vector<KGNodeData>* GetFlattenedTreePointer() {return &fFlattenedTree;};

        void GetIdentitySetNodeIDs(std::vector<int>* node_ids) const {*node_ids = fIdentitySetNodeIDs;};
        void SetIdentitySetNodeIDs(const std::vector<int>* node_ids) {fIdentitySetNodeIDs = *node_ids;};
        const std::vector<int>* GetIdentitySetNodeIDPointer() const {return &fIdentitySetNodeIDs;};
        std::vector<int>* GetIdentitySetNodeIDPointer() {return &fIdentitySetNodeIDs;};

        void GetIdentitySets(std::vector< KGIdentitySet* >* id_sets) const {*id_sets = fIdentitySets;};
        void SetIdentitySets(const std::vector< KGIdentitySet* >* id_sets) {fIdentitySets = *id_sets;};
        const std::vector< KGIdentitySet* >* GetIdentitySetPointer() const {return &fIdentitySets;};
        std::vector< KGIdentitySet* >* GetIdentitySetPointer() {return &fIdentitySets;};

        void GetCubeNodeIDs(std::vector<int>* node_ids) const {*node_ids = fCubeNodeIDs;};
        void SetCubeNodeIDs(const std::vector<int>* node_ids) {fCubeNodeIDs = *node_ids;};
        const std::vector<int>* GetCubeNodeIDPointer() const {return &fCubeNodeIDs;};
        std::vector<int>* GetCubeNodeIDPointer() {return &fCubeNodeIDs;};

        void GetCubes(std::vector< KGCube<KGMESH_DIM>* >* cubes) const {*cubes = fCubes;};
        void SetCubes(const std::vector< KGCube<KGMESH_DIM>* >* cubes) {fCubes = *cubes;};
        const std::vector< KGCube<KGMESH_DIM>* >* GetCubePointer() const {return &fCubes;};
        std::vector< KGCube<KGMESH_DIM>* >* GetCubePointer() {return &fCubes;};

        void SetMaximumOctreeDepth(unsigned int d){fMaxDepth = d;};
        unsigned int GetMaximumOctreeDepth() const {return fMaxDepth;};

        void SetSpecifyMaximumOctreeDepth(unsigned int specify)
        {
            fSpecifyMaxDepth = 0; if(specify){fSpecifyMaxDepth = 1;};
        };
        unsigned int GetSpecifyMaximumOctreeDepth() const {if(fSpecifyMaxDepth == 1){return true;}; return false;};

        void SetSpatialResolution(double r){fSpatialResolution = r;};
        double GetSpatialResolution() const { return fSpatialResolution;};

        void SetSpecifySpatialResolution(unsigned int specify)
        {
            fSpecifyResolution = 0; if(specify){fSpecifyResolution = 1;};
        };
        unsigned int GetSpecifySpatialResolution() const {if(fSpecifyResolution== 1){return true;}; return false;};

        void SetNumberOfAllowedElements(unsigned int n){fNAllowedElements = n; fSpecifyAllowedElements = true;};
        unsigned int GetNumberOfAllowedElements() const {return fNAllowedElements;};

        void SetSpecifyNumberOfAllowedElements(unsigned int specify)
        {
            fSpecifyAllowedElements= 0; if(specify){fSpecifyAllowedElements = 1;};
        };
        unsigned int GetSpecifyNumberOfAllowedElements() const  {if( fSpecifyAllowedElements == 1){return true;}; return false;};

    private:

        //octree construction parameters
        unsigned int fMaxDepth;
        unsigned int fSpecifyMaxDepth;

        double fSpatialResolution;
        unsigned int fSpecifyResolution;

        unsigned int fNAllowedElements;
        unsigned int fSpecifyAllowedElements;

        //storage for the tree structure (parent to child links)
        std::string fTreeID;
        unsigned int fNNodes;
        std::vector<KGNodeData> fFlattenedTree;

        //list of mesh elements mappedd to each node
        std::vector< int > fIdentitySetNodeIDs;
        std::vector< KGIdentitySet* > fIdentitySets;

        //list of cubes mapped to each node
        std::vector< int > fCubeNodeIDs;
        std::vector< KGCube<KGMESH_DIM>* > fCubes;
};




template <typename Stream>
Stream& operator>>(Stream& s, KSNavOctreeData& aData)
{
    s.PreStreamInAction(aData);
    //
    unsigned int max_depth;
    s >> max_depth;
    aData.SetMaximumOctreeDepth(max_depth);

    unsigned int specify_max_depth;
    s >> specify_max_depth;
    aData.SetSpecifyMaximumOctreeDepth(specify_max_depth);

    double spatial_res;
    s >> spatial_res;
    aData.SetSpatialResolution(spatial_res);

    unsigned int specify_res;
    s >> specify_res;
    aData.SetSpecifySpatialResolution(specify_res);

    unsigned int n_allowed;
    s >> n_allowed;
    aData.SetNumberOfAllowedElements(n_allowed);

    unsigned int specify_allowed;
    s >> specify_allowed;
    aData.SetSpecifyNumberOfAllowedElements(specify_allowed);

    unsigned int n_nodes;
    s >> n_nodes;
    aData.SetNumberOfTreeNodes(n_nodes);

    std::vector<KGNodeData>* flattened_tree = aData.GetFlattenedTreePointer();
    unsigned int n_flattened_nodes;
    s >> n_flattened_nodes;
    flattened_tree->resize(0);
    flattened_tree->reserve(n_flattened_nodes);
    for(unsigned int i=0; i < n_flattened_nodes; i++)
    {
        KGNodeData node_data;
        s >> node_data;
        flattened_tree->push_back(node_data);
    }


    std::vector<int>* id_set_node_ids = aData.GetIdentitySetNodeIDPointer();
    unsigned int id_set_node_ids_size;
    s >> id_set_node_ids_size;
    id_set_node_ids->resize(0);
    id_set_node_ids->reserve(id_set_node_ids_size);
    for(unsigned int i=0; i < id_set_node_ids_size; i++)
    {
        unsigned int id;
        s >> id;
        id_set_node_ids->push_back(id);
    }

    std::vector< KGIdentitySet* >* id_set = aData.GetIdentitySetPointer();
    unsigned int id_set_size;
    s >> id_set_size;
    id_set->resize(0);
    id_set->reserve(id_set_size);
    for(unsigned int i=0; i < id_set_size; i++)
    {
        KGIdentitySet* set = new KGIdentitySet();
        s >> *set;
        id_set->push_back(set);
    }

    std::vector<int>* cube_node_ids = aData.GetCubeNodeIDPointer();
    unsigned int cube_node_ids_size;
    s >> cube_node_ids_size;
    cube_node_ids->resize(0);
    cube_node_ids->reserve(cube_node_ids_size);
    for(unsigned int i=0; i < cube_node_ids_size; i++)
    {
        unsigned int id;
        s >> id;
        cube_node_ids->push_back(id);
    }

    std::vector< KGCube<KGMESH_DIM>* >* cubes = aData.GetCubePointer();
    unsigned int cube_size;
    s >> cube_size;
    cubes->resize(0);
    cubes->reserve(cube_size);
    for(unsigned int i=0; i < cube_size; i++)
    {
        KGCube<KGMESH_DIM>* cube = new KGCube<KGMESH_DIM>();
        s >> *cube;
        cubes->push_back(cube);
    }

    s.PostStreamInAction(aData);
    return s;
}

template <typename Stream>
Stream& operator<<(Stream& s,const KSNavOctreeData& aData)
{
    s.PreStreamOutAction(aData);

    s << aData.GetMaximumOctreeDepth();
    s << aData.GetSpecifyMaximumOctreeDepth();
    s << aData.GetSpatialResolution();
    s << aData.GetSpecifySpatialResolution();
    s << aData.GetNumberOfAllowedElements();
    s << aData.GetSpecifyNumberOfAllowedElements();
    s << aData.GetNumberOfTreeNodes();

    const std::vector<KGNodeData>* flattened_tree = aData.GetFlattenedTreePointer();
    unsigned int n_flattened_nodes = flattened_tree->size();
    s << n_flattened_nodes;
    for(unsigned int i=0; i< n_flattened_nodes; i++)
    {
        s << (*flattened_tree)[i];
    }

    const std::vector<int>* id_set_node_ids = aData.GetIdentitySetNodeIDPointer();
    unsigned int id_set_node_ids_size = id_set_node_ids->size();
    s << id_set_node_ids_size;
    for(unsigned int i=0; i < id_set_node_ids_size; i++)
    {
        s << (*id_set_node_ids)[i];
    }

    const std::vector< KGIdentitySet* >* id_set = aData.GetIdentitySetPointer();
    unsigned int id_set_size = id_set->size();
    s << id_set_size;
    for(unsigned int i=0; i < id_set_size; i++)
    {
        s << *( (*id_set)[i] );
    }

    const std::vector<int>* cube_node_ids = aData.GetCubeNodeIDPointer();
    unsigned int cube_node_ids_size = cube_node_ids->size();
    s << cube_node_ids_size;
    for(unsigned int i=0; i < cube_node_ids_size; i++)
    {
        s << (*cube_node_ids)[i];
    }

    const std::vector< KGCube<KGMESH_DIM>* >* cubes = aData.GetCubePointer();
    unsigned int cube_size = cubes->size();
    s << cube_size;
    for(unsigned int i=0; i < cube_size; i++)
    {
        s << *( (*cubes)[i] );
    }

    s.PostStreamOutAction(aData);

    return s;
}


}

#endif /* KSNavOctreeData_H__ */
