#ifndef KFMExternalIdentitySetSorter_HH__
#define KFMExternalIdentitySetSorter_HH__

#include "KFMExternalIdentitySet.hh"
#include "KFMNode.hh"
#include "KFMNodeActor.hh"
#include "KFMObjectRetriever.hh"

namespace KEMField
{

/*
*
*@file KFMExternalIdentitySetSorter.hh
*@class KFMExternalIdentitySetSorter
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Mar  3 10:01:48 EST 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template< typename ObjectTypeList>
class KFMExternalIdentitySetSorter: public KFMNodeActor< KFMNode<ObjectTypeList> >
{
    public:
        KFMExternalIdentitySetSorter(){};
        virtual ~KFMExternalIdentitySetSorter(){};

        virtual void ApplyAction( KFMNode<ObjectTypeList>* node)
        {
            KFMExternalIdentitySet* set = KFMObjectRetriever<ObjectTypeList, KFMExternalIdentitySet>::GetNodeObject(node);
            if(set != NULL)
            {
                set->Sort();
            }
        }

    private:
};


}

#endif /* KFMExternalIdentitySetSorter_H__ */
