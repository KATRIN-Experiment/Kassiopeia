#ifndef __KFMNonEmptyIdentitySetActor_H__
#define __KFMNonEmptyIdentitySetActor_H__

#include "KFMIdentitySet.hh"
#include "KFMInspectingActor.hh"
#include "KFMNode.hh"
#include "KFMObjectRetriever.hh"

namespace KEMField
{

/**
*
*@file KFMNonEmptyIdentitySetActor.hh
*@class KFMNonEmptyIdentitySetActor
*@brief
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Jul 14 11:05:31 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<typename ObjectTypeList> class KFMNonEmptyIdentitySetActor : public KFMInspectingActor<KFMNode<ObjectTypeList>>
{
  public:
    KFMNonEmptyIdentitySetActor(){};
    ~KFMNonEmptyIdentitySetActor() override{};

    bool ConditionIsSatisfied(KFMNode<ObjectTypeList>* node) override
    {
        if (node != nullptr) {
            KFMIdentitySet* set = KFMObjectRetriever<ObjectTypeList, KFMIdentitySet>::GetNodeObject(node);
            if (set != nullptr) {
                if (set->GetSize() != 0) {
                    return true;
                }
            }
        }
        return false;
    }


  protected:
    /* data */
};


}  // namespace KEMField

#endif /* __KFMNonEmptyIdentitySetActor_H__ */
