#ifndef KFMNearbyElementCounter_HH__
#define KFMNearbyElementCounter_HH__


#include <vector>

#include "KFMNode.hh"
#include "KFMNodeActor.hh"
#include "KFMObjectRetriever.hh"
#include "KFMIdentitySet.hh"

#include "KFMCubicSpaceNodeNeighborFinder.hh"

namespace KEMField
{

/*
*
*@file KFMNearbyElementCounter.hh
*@class KFMNearbyElementCounter
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Nov  6 16:55:23 EST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ObjectTypeList, unsigned int NDIM>
class KFMNearbyElementCounter: public KFMNodeActor< KFMNode<ObjectTypeList> >
{
    public:
        KFMNearbyElementCounter(){};
        virtual ~KFMNearbyElementCounter(){};

        void SetNeighborOrder(int order){fOrder = std::fabs(order);};

        virtual void ApplyAction(KFMNode<ObjectTypeList>* node)
        {
            fNumberOfNearbyElements = 0;

            if(node != NULL)
            {
                KFMCubicSpaceNodeNeighborFinder<NDIM, ObjectTypeList>::GetAllNeighbors(node, fOrder, &fNeighbors);

                for(unsigned int i=0; i<fNeighbors.size(); i++)
                {
                    KFMIdentitySet* id_set = KFMObjectRetriever<ObjectTypeList , KFMIdentitySet >::GetNodeObject(fNeighbors[i]);
                    if(id_set != NULL)
                    {
                        fNumberOfNearbyElements += id_set->GetSize();
                    }
                }
            }
        }

        unsigned int GetNumberOfNearbyElements(){return fNumberOfNearbyElements;};

    private:

        unsigned int fOrder;
        std::vector< KFMNode<ObjectTypeList>* > fNeighbors;

        unsigned int fNumberOfNearbyElements;
};


}


#endif /* KFMNearbyElementCounter_H__ */
