#ifndef KFMCubicSpaceNodeNeighborFinder_HH__
#define KFMCubicSpaceNodeNeighborFinder_HH__

#include "KFMArrayMath.hh"
#include "KFMNode.hh"

#include "KFMCubicSpaceTreeProperties.hh"

namespace KEMField{

/**
*
*@file KFMCubicSpaceNodeNeighborFinder.hh
*@class KFMCubicSpaceNodeNeighborFinder
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Jul 25 21:28:02 EDT 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<unsigned int NDIM, typename ObjectTypeList>
class KFMCubicSpaceNodeNeighborFinder
{
    public:
        KFMCubicSpaceNodeNeighborFinder(){};
        virtual ~KFMCubicSpaceNodeNeighborFinder(){};

        //get all neighbors that are up to 'order' positions away from the target node
        static void
        GetAllNeighbors(KFMNode<ObjectTypeList>* target_node, unsigned int order, std::vector< KFMNode<ObjectTypeList>* >* neighbors)
        {
            neighbors->clear();

            //number of positions in each direction a neighbor might be found
            int stride = 2*order + 1;
            unsigned int dim_size[NDIM]; //here dim_size is the dimensionality of the neighbor array, not the tree!
            unsigned int temp_index[NDIM];
            int index[NDIM];

            for(unsigned int i=0; i<NDIM; i++)
            {
                dim_size[i] = stride;
            }

            //max possible number of neighbors for this order
            unsigned int list_size = KFMArrayMath::TotalArraySize<NDIM>(dim_size);
            neighbors->resize(list_size);
            for(unsigned int i=0; i<list_size; i++)
            {
                neighbors->at(i) = NULL;
            }

            for(unsigned int n=0; n<list_size; n++)
            {
                KFMArrayMath::RowMajorIndexFromOffset<NDIM>(n, dim_size, temp_index);
                for(unsigned int i=0; i<NDIM; i++)
                {
                    index[i] = (int)temp_index[i] - order; //neighbor indices indicate relative position, so they can be negative
                }
                neighbors->at(n) = KFMCubicSpaceNodeNeighborFinder<NDIM, ObjectTypeList>::GetNeighbor(target_node, index);
            }

        }

        static KFMNode<ObjectTypeList>*
        GetNeighbor(KFMNode<ObjectTypeList>* target_node, int* index)
        {
            //index is the relative spatial position (in number of cubes) of the neighbor we are looking for

            //check to make sure the coordinates of the neighbor we are looking
            //for do not correspond to the original node
            bool isSelf = true;
            for(unsigned int i=0; i<NDIM; i++)
            {
                if(index[i] != 0){isSelf = false;}
            }
            if(isSelf){return target_node;};

            //space for calculations
            int abs_coord[NDIM];
            int div_coord[NDIM];
            unsigned int mod_coord[NDIM];
            unsigned int target_coord[NDIM];
            unsigned int dim_size[NDIM];

            //get this node's parent
            KFMNode<ObjectTypeList>* parent = target_node->GetParent();

            if(parent != NULL)
            {
                //get the dimensionality of the divisions at this tree level
                if(parent->GetLevel() == 0)
                {
                    //we are at the top level
                    KFMObjectRetriever<ObjectTypeList, KFMCubicSpaceTreeProperties<NDIM> >::GetNodeObject(target_node)->GetTopLevelDimensions(dim_size);
                }
                else
                {
                    //all other tree levels
                    KFMObjectRetriever<ObjectTypeList, KFMCubicSpaceTreeProperties<NDIM> >::GetNodeObject(target_node)->GetDimensions(dim_size);
                }

                //get the target nodes storage index in its parents list
                unsigned int target_storage_index = target_node->GetIndex();

                //compute its spatial indices from the storage index
                KFMArrayMath::RowMajorIndexFromOffset<NDIM>(target_storage_index, dim_size, target_coord);

                for(unsigned int i=0; i<NDIM; i++)
                {
                    abs_coord[i] = index[i] + target_coord[i];
                    div_coord[i] = (int)(std::floor( ((double)abs_coord[i]) / ( (double)dim_size[i] ) ) );
                    mod_coord[i] = KFMArrayMath::Modulus(abs_coord[i], dim_size[i]);
                }

                KFMNode<ObjectTypeList>* parent_neighbor;
                parent_neighbor = KFMCubicSpaceNodeNeighborFinder<NDIM, ObjectTypeList>::GetNeighbor(parent, div_coord);

                if(parent_neighbor != NULL)
                {
                    return parent_neighbor->GetChild( KFMArrayMath::OffsetFromRowMajorIndex<NDIM>(dim_size, mod_coord) );
                }
                else
                {
                    return NULL;
                }
            }
            else
            {
                return NULL;
            }
        }

    private:


};

}


#endif /* KFMCubicSpaceNodeNeighborFinder_H__ */
