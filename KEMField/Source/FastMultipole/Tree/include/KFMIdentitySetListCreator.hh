#ifndef __KFMIdentitySetListCreator_H__
#define __KFMIdentitySetListCreator_H__

#include "KFMNode.hh"
#include "KFMNodeActor.hh"
#include "KFMIdentitySet.hh"
#include "KFMIdentitySetList.hh"

namespace KEMField
{

/**
*
*@file KFMIdentitySetListCreator.hh
*@class KFMIdentitySetListCreator
*@brief
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Sep 18 13:44:18 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template< typename ObjectTypeList >
class KFMIdentitySetListCreator: public KFMNodeActor< KFMNode<ObjectTypeList> >
{
    public:
        KFMIdentitySetListCreator()
        {
            fZeroMaskSize = 0;
            fMaxSize = 0;
        };
        virtual ~KFMIdentitySetListCreator(){};

        void SetZeroMaskSize(unsigned int zmask){fZeroMaskSize = zmask;};
        unsigned int GetMaxExternalIDSetSize() const {return fMaxSize;};

        virtual void ApplyAction( KFMNode< ObjectTypeList >* node)
        {
            if(node != NULL )
            {
                KFMIdentitySetList* set_list = new KFMIdentitySetList(); //create a new id set list

                //now we will visit this node's immediate neighbors (at the same tree level) and if they own elements, we will collect
                //them and add them to this node's id set list
                bool create_new_id_set_list = false;

                //get neighbor nodes
                fNodeNeighborList.clear();
                KFMCubicSpaceNodeNeighborFinder<3, ObjectTypeList>::GetAllNeighbors(node, fZeroMaskSize, &fNodeNeighborList);

                for(unsigned int j=0; j<fNodeNeighborList.size(); j++)
                {
                    KFMIdentitySet* id_set = NULL;
                    if(fNodeNeighborList[j] != NULL)
                    {
                        id_set =  KFMObjectRetriever<ObjectTypeList, KFMIdentitySet>::GetNodeObject( fNodeNeighborList[j]);
                        if(id_set != NULL)
                        {
                            if(id_set->GetSize() != 0)
                            {
                                create_new_id_set_list = true;
                                set_list->AddIDSet(id_set);
                            }
                        }
                    }
                }

                if(create_new_id_set_list)
                {
                    //assign the newly created id set list to this node
                    KFMObjectRetriever<ObjectTypeList, KFMIdentitySetList>::SetNodeObject(set_list, node);
                    //compute max size
                    unsigned int set_size = set_list->GetTotalSize();
                    if(fMaxSize < set_size){fMaxSize = set_size;};
                }
                else
                {
                    //delete the current superfluous id set list
                    delete set_list;
                    //set this node's id set list to NULL
                    KFMObjectRetriever<ObjectTypeList, KFMIdentitySetList>::SetNodeObject(NULL, node);
                }
            }
        }


    protected:

        unsigned int fZeroMaskSize;
        std::vector< KFMNode<ObjectTypeList>* > fNodeNeighborList;
        unsigned int fMaxSize;

};

}

#endif /* __KFMIdentitySetListCreator_H__ */
