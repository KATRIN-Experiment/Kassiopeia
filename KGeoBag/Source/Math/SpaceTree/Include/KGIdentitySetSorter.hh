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

template<typename ObjectTypeList> class KGIdentitySetSorter : public KGNodeActor<KGNode<ObjectTypeList>>
{
  public:
    KGIdentitySetSorter() = default;
    ;
    virtual ~KGIdentitySetSorter() = default;
    ;

    virtual void ApplyAction(KGNode<ObjectTypeList>* node)
    {
        KGIdentitySet* set = KGObjectRetriever<ObjectTypeList, KGIdentitySet>::GetNodeObject(node);
        if (set != nullptr) {
            set->Sort();
        }
    }

  private:
};


}  // namespace KGeoBag

#endif /* KGIdentitySetSorter_H__ */
