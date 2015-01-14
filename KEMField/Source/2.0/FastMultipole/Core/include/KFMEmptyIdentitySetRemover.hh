#ifndef KFMEmptyIdentitySetRemover_HH__
#define KFMEmptyIdentitySetRemover_HH__

#include "KFMNode.hh"
#include "KFMIdentitySet.hh"
#include "KFMExternalIdentitySet.hh"
#include "KFMNodeActor.hh"
#include "KFMObjectRetriever.hh"

namespace KEMField
{

/*
*
*@file KFMEmptyIdentitySetRemover.hh
*@class KFMEmptyIdentitySetRemover
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Apr  3 12:02:46 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template< typename ObjectTypeList >
class KFMEmptyIdentitySetRemover: public KFMNodeActor< KFMNode<ObjectTypeList> >
{
    public:
        KFMEmptyIdentitySetRemover(){};
        virtual ~KFMEmptyIdentitySetRemover(){};


        virtual void ApplyAction( KFMNode<ObjectTypeList>* node)
        {
            if(node != NULL)
            {
                KFMIdentitySet* set = KFMObjectRetriever<ObjectTypeList, KFMIdentitySet>::GetNodeObject(node);

                if(set != NULL)
                {
                    if(set->GetSize() == 0)
                    {
                        delete set;
                        KFMObjectRetriever<ObjectTypeList, KFMIdentitySet>::SetNodeObject(NULL, node);
                    }
                }

                KFMExternalIdentitySet* eset = KFMObjectRetriever<ObjectTypeList, KFMExternalIdentitySet>::GetNodeObject(node);

                if(eset != NULL)
                {
                    if(eset->GetSize() == 0)
                    {
                        delete eset;
                        KFMObjectRetriever<ObjectTypeList, KFMExternalIdentitySet>::SetNodeObject(NULL, node);
                    }
                }

            }
        }

    private:
};

}

#endif /* KFMEmptyIdentitySetRemover_H__ */
