#ifndef KFMCubicSpaceTreeProperties_HH__
#define KFMCubicSpaceTreeProperties_HH__

#include <string>

namespace KEMField{

/**
*
*@file KFMCubicSpaceTreeProperties.hh
*@class KFMCubicSpaceTreeProperties
*@brief simple container for a small set of parameters that all nodes in a tree share
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
* J. Barrett (barrettj@mit.edu) First Version
*
*/

template< unsigned int NDIM >
class KFMCubicSpaceTreeProperties
{
    public:

        KFMCubicSpaceTreeProperties():fTreeID(""),fCubicNeighborOrder(0),fCurrentMaxUniqueID(0)
        {
            for(unsigned int i=0; i<NDIM; i++)
            {
                fDimSize[i] = 0;
            };

            fCubicNeighborOrder = 0;
            fMaxTreeDepth = 0;
        };

        virtual ~KFMCubicSpaceTreeProperties(){};

    public:

        //unique id for the entire tree
        void SetTreeID(std::string tree_id){fTreeID = tree_id;};
        std::string GetTreeID() const {return fTreeID;};

        //max depth of the tree
        void SetMaxTreeDepth(unsigned int depth){fMaxTreeDepth = depth;};
        unsigned int GetMaxTreeDepth() const {return fMaxTreeDepth;};

        //nodes which are less than or equal this number of nodes away are considered neighbors
        void SetCubicNeighborOrder(unsigned int cn_order){fCubicNeighborOrder = cn_order;};
        unsigned int GetCubicNeighborOrder() const {return fCubicNeighborOrder;};

        //get the number of dimension of the divisions of the tree
        unsigned int GetNDimensions() const {return NDIM;};

        //get/set the size of each dimension
        const unsigned int* GetDimensions() const {return fDimSize;};
        void GetDimensions(unsigned int* dim_size) const
        {
            for(unsigned int i=0; i<NDIM; i++)
            {
                dim_size[i] = fDimSize[i];
            }
        }
        void SetDimensions(const unsigned int* dim_size)
        {
            for(unsigned int i=0; i<NDIM; i++)
            {
                fDimSize[i] = dim_size[i];
            }
        }

        int RegisterNode()
        {
            int id = fCurrentMaxUniqueID;
            fCurrentMaxUniqueID++;
            return id;
        }


        //the condition for 'primacy' of any particular node is determined by the
        //actor which changes this condition, it does not need to remain a constant
        //property throughout useage of the tree

//        //set the status of a particular node
//        void SetNodePrimaryStatus(unsigned int node_id, bool status)
//        {
//            if(node_id >= fPrimaryNodeStatus.size())
//            {
//                fPrimaryNodeStatus.resize(node_id + 1);
//            }

//            fPrimaryNodeStatus[node_id] = status;
//        }

//        //get the status of a particular node
//        bool GetNodePrimaryStatus(unsigned int node_id)
//        {
//            if(node_id >= fPrimaryNodeStatus.size())
//            {
//                return false;
//            }
//            else
//            {
//                return fPrimaryNodeStatus[node_id];
//            }
//        }

        unsigned int GetNNodes() const {return fCurrentMaxUniqueID;};

    private:

        unsigned int fDimSize[NDIM]; //the number divisions in each dimension of the sub-division

        std::string fTreeID; //the id pertaining to the entire tree

        unsigned int fCubicNeighborOrder;

        unsigned int fMaxTreeDepth;

        unsigned int fCurrentMaxUniqueID;

//        //indicates whether a specific node is 'primary'
//        //the node id is the index in this list
//        std::vector< bool > fPrimaryNodeStatus;


};


}//end of KEMField


#endif /* KFMCubicSpaceTreeProperties_H__ */
