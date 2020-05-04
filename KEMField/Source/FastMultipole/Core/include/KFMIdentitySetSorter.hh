#ifndef KFMIdentitySetSorter_HH__
#define KFMIdentitySetSorter_HH__

#include "KFMIdentitySet.hh"
#include "KFMNode.hh"
#include "KFMNodeActor.hh"
#include "KFMObjectRetriever.hh"

namespace KEMField
{

/*
*
*@file KFMIdentitySetSorter.hh
*@class KFMIdentitySetSorter
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Mar  3 10:01:48 EST 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ObjectTypeList> class KFMIdentitySetSorter : public KFMNodeActor<KFMNode<ObjectTypeList>>
{
  public:
    KFMIdentitySetSorter(){};
    ~KFMIdentitySetSorter() override{};

    void ApplyAction(KFMNode<ObjectTypeList>* node) override
    {
        KFMIdentitySet* set = KFMObjectRetriever<ObjectTypeList, KFMIdentitySet>::GetNodeObject(node);
        if (set != nullptr) {
            set->Sort();
        }
    }

  private:
};


}  // namespace KEMField

#endif /* KFMIdentitySetSorter_H__ */
