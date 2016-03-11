#include "KGSubdivisionCondition.hh"
#include "KGInsertionCondition.hh"

namespace KGeoBag
{

bool
KGSubdivisionCondition::ConditionIsSatisfied(KGMeshNavigationNode* node)
{
    //first get the tree properties associated with this node
    KGSpaceTreeProperties<KGMESH_DIM>* tree_prop = KGObjectRetriever<KGMeshNavigationNodeObjects, KGSpaceTreeProperties<KGMESH_DIM> >::GetNodeObject(node);
    unsigned int max_depth = tree_prop->GetMaxTreeDepth();
    unsigned int level = node->GetLevel();
    fDimSize = tree_prop->GetDimensions();

    if(level < max_depth)
    {
        //get the list of mesh element id's
        KGIdentitySet* elem_list = KGObjectRetriever<KGMeshNavigationNodeObjects, KGIdentitySet >::GetNodeObject(node);
        if(elem_list->GetSize() < fNAllowedElements)
        {
            //number of elements in this node is not greater than allowed
            //so don't bother subdividing it
            return false;
        }

        //now we are going to compare the size of the elements to
        //the side of the children that would be created
        //if there are no elements that are smaller than a child size
        //then we do not divide this node

        //get the geometric properties of this node
        KGCube<KGMESH_DIM>* cube = KGObjectRetriever<KGMeshNavigationNodeObjects, KGCube<KGMESH_DIM> >::GetNodeObject(node);
        fLength = cube->GetLength();
        //we make the assumption that the dimensions of each division have the same size (valid for cubes)
        double division = fDimSize[0];
        fLength = fLength/division; //length of a child node

        //now look at the element sizes
        std::vector<unsigned int> elem_id_list;
        elem_list->GetIDs(&elem_id_list);
        unsigned int list_size = elem_list->GetSize();
        for(unsigned int i=0; i<list_size; i++)
        {
            KGBall<KGMESH_DIM> bball = fContainer->GetElementBoundingBall(elem_id_list[i]);
            if(  2.0*bball.GetRadius() < fLength)
            {
                //we have at least one element which has a size smaller than
                //a child node, so we can subdivide this node
                return true;
            }
        }

        //no elements which are small enough, don't bother dividing
        return false;
    }
    else
    {
        return false;
    }

}

}
