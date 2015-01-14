#ifndef KGIdentitySetSorter_HH__
#define KGIdentitySetSorter_HH__

#include "KGIdentitySet.hh"
#include "KGNode.hh"
#include "KGNodeActor.hh"
#include "KGObjectRetriever.hh"

namespace KGeoBag
{

/*
*
*@file KGIdentitySetSorter.hh
*@class KGIdentitySetSorter
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Mar  3 10:01:48 EST 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template< typename ObjectTypeList>
class KGIdentitySetSorter: public KGNodeActor< KGNode<ObjectTypeList> >
{
    public:
        KGIdentitySetSorter(){};
        virtual ~KGIdentitySetSorter(){};

        virtual void ApplyAction( KGNode<ObjectTypeList>* node)
        {
            KGIdentitySet* set = KGObjectRetriever<ObjectTypeList, KGIdentitySet>::GetNodeObject(node);
            if(set != NULL)
            {
                set->Sort();
            }
        }

    private:
};


}

#endif /* KGIdentitySetSorter_H__ */
