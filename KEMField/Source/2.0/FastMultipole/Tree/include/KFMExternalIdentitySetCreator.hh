#ifndef KFMExternalIdentitySetCreator_HH__
#define KFMExternalIdentitySetCreator_HH__

#include "KFMNode.hh"
#include "KFMNodeActor.hh"
#include "KFMIdentitySet.hh"
#include "KFMExternalIdentitySet.hh"
#include "KFMIdentitySetMerger.hh"

namespace KEMField
{

/*
*
*@file KFMExternalIdentitySetCreator.hh
*@class KFMExternalIdentitySetCreator
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Jan 23 16:59:56 EST 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template< typename ObjectTypeList >
class KFMExternalIdentitySetCreator: public KFMNodeActor< KFMNode<ObjectTypeList> >
{
    public:

        KFMExternalIdentitySetCreator()
        {
            fZeroMaskSize = 0;
            fMaxSize = 0;
        };
        virtual ~KFMExternalIdentitySetCreator(){};

        void SetZeroMaskSize(unsigned int zmask){fZeroMaskSize = zmask;};

        unsigned int GetMaxExternalIDSetSize() const {return fMaxSize;};

        virtual void ApplyAction( KFMNode< ObjectTypeList >* node)
        {
            if(node != NULL )
            {
                if(node->GetParent() == NULL )
                {
                    //this is the root node, its external id set should only contain elements that it itself owns
                    KFMIdentitySet* root_id_set = KFMObjectRetriever<ObjectTypeList, KFMIdentitySet>::GetNodeObject(node);

                    if(root_id_set != NULL)
                    {
                        if(root_id_set->GetSize() != 0)
                        {
                            root_id_set->GetIDs(&ids);

                            KFMExternalIdentitySet* root_eid_set = new KFMExternalIdentitySet();
                            root_eid_set->SetIDs(&ids);
                            KFMObjectRetriever<ObjectTypeList, KFMExternalIdentitySet>::SetNodeObject(root_eid_set, node);

                            if(fMaxSize < root_eid_set->GetSize())
                            {
                                fMaxSize = root_eid_set->GetSize();
                            }
                        }
                    }
                    return;
                }

                //first we assign this node's external id set to point to it's parents ID set
                KFMExternalIdentitySet* parent_eid_set =  KFMObjectRetriever<ObjectTypeList, KFMExternalIdentitySet>::GetNodeObject( node->GetParent() );
                KFMObjectRetriever<ObjectTypeList, KFMExternalIdentitySet>::SetNodeObject(parent_eid_set, node);

                //now we will visit this node's immediate neighbors (at the same tree level) and if they own elements, we will collect
                //them and add them to this node's external id list
                fIDCollector.Clear();
                fNodeNeighborList.clear();
                KFMCubicSpaceNodeNeighborFinder<3, ObjectTypeList>::GetAllNeighbors(node, fZeroMaskSize, &fNodeNeighborList);

                for(unsigned int j=0; j<fNodeNeighborList.size(); j++)
                {
                    fIDCollector.ApplyAction( fNodeNeighborList[j] );
                }

                //now check if we have collected any additional ids, we give this node a new external id set
                //otherwise leave it pointing to its parents id set
                if( fIDCollector.GetIDSet()->GetSize() != 0 )
                {

                    parent_ids.clear();
                    //if parents set is not empty, get those id's first
                    if(parent_eid_set != NULL)
                    {
                        if(parent_eid_set->GetSize() != 0)
                        {
                            parent_eid_set->GetIDs(&parent_ids);
                        }
                    }

                    KFMExternalIdentitySet* set = new KFMExternalIdentitySet();
                    fIDCollector.GetIDSet()->GetIDs(&ids);

                    //append the parent's external id's
                    ids.insert( ids.end(), parent_ids.begin(), parent_ids.end() );
                    set->SetIDs(&ids);
                    KFMObjectRetriever<ObjectTypeList, KFMExternalIdentitySet>::SetNodeObject(set, node);

                    if(fMaxSize < set->GetSize())
                    {
                        fMaxSize = set->GetSize();
                    }

                }
            }
        }

    private:

        unsigned int fZeroMaskSize;
        std::vector< KFMNode<ObjectTypeList>* > fNodeNeighborList;

        KFMIdentitySetMerger< ObjectTypeList > fIDCollector;
        std::vector<unsigned int> ids;
        std::vector<unsigned int> parent_ids;

        unsigned int fMaxSize;

};


}//end KEMField namespace

#endif /* KFMExternalIdentitySetCreator_H__ */
