#include "KGNavigableMeshElementSorter.hh"

namespace KGeoBag
{

void
KGNavigableMeshElementSorter::ApplyAction( KGMeshNavigationNode* node)
{
    //in this function we distribute the mesh located in a node
    //to its children if the children intersect the elements
    if(node->HasChildren())
    {
        //get the list of element id's from the node
        KGIdentitySet* element_list = KGObjectRetriever<KGMeshNavigationNodeObjects, KGIdentitySet >::GetNodeObject(node);
        std::vector<unsigned int> original_id_list;
        element_list->GetIDs(&original_id_list);

        //now we iterate over the node's list of elements
        //and decide which ones should be pushed downwards to a child
        const KGNavigableMeshElement* mesh_element;
        unsigned int n_children = node->GetNChildren();
        std::vector< std::vector<unsigned int> > child_id_list; //id list for each child
        std::vector< unsigned int > updated_id_list; //new id list for the parent after redistribution
        std::vector< unsigned int > temp_list; //temp list for elements still in parent
        child_id_list.resize(n_children);
        KGCube<KGMESH_DIM>* child_cube;
        KGMeshNavigationNode* child;

        updated_id_list = original_id_list;
        temp_list.clear();
        for(unsigned int j=0; j<n_children; j++)
        {
            child = node->GetChild(j);
            child_cube = KGObjectRetriever<KGMeshNavigationNodeObjects, KGCube<KGMESH_DIM> >::GetNodeObject(child);
            KGPoint<KGMESH_DIM> cube_enter = child_cube->GetCenter();
            double child_len = child_cube->GetLength();
            double cube_radius = std::sqrt(3.0*(child_len/2.0)*(child_len/2.0));

            unsigned int list_size = updated_id_list.size();
            for(unsigned int i=0; i<list_size; i++)
            {
                mesh_element = fContainer->GetElement(updated_id_list[i]);
                bool parent_owns_element = true;

                KGBall<KGMESH_DIM> bball = fContainer->GetElementBoundingBall(updated_id_list[i]);
                KGPoint<KGMESH_DIM> ball_center = bball.GetCenter();
                double dist = (ball_center - cube_enter).Magnitude();

                if(dist <= bball.GetRadius() + cube_radius)
                {
                    if(fCondition.ElementIntersectsCube(mesh_element, child_cube))
                    {
                        child_id_list[j].push_back(updated_id_list[i]);

                        if(fCondition.ElementEnclosedByCube(mesh_element, child_cube))
                        {
                            //element is enclosed by this child node,
                            //so we can remove it from the parent's list, so that
                            //we don't have to compare it to any other children
                            parent_owns_element = false;
                        }
                    }
                }

                if(parent_owns_element)
                {
                    temp_list.push_back(updated_id_list[i]);
                }
            }

            updated_id_list = temp_list;
            temp_list.clear();

        }

        //since the parent node is no longer a leaf node, we remove all of its element ids
        updated_id_list.clear();
        element_list->SetIDs(&updated_id_list);

        //update/create the list for each child
        KGIdentitySet* child_list;
        for(unsigned int j=0; j<n_children; j++)
        {
            child = node->GetChild(j);
            child_list = KGObjectRetriever<KGMeshNavigationNodeObjects,  KGIdentitySet >::GetNodeObject(child);

            delete child_list; //delete it if it already exists

            child_list = new KGIdentitySet();
            child_list->SetIDs( &(child_id_list[j]) );

            KGObjectRetriever<KGMeshNavigationNodeObjects, KGIdentitySet  >::SetNodeObject(child_list, child);
        }

    }

}


}
