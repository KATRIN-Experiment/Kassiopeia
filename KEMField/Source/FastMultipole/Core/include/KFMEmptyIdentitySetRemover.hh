#ifndef KFMEmptyIdentitySetRemover_HH__
#define KFMEmptyIdentitySetRemover_HH__

#include "KFMIdentitySet.hh"
#include "KFMNode.hh"
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

template<typename ObjectTypeList> class KFMEmptyIdentitySetRemover : public KFMNodeActor<KFMNode<ObjectTypeList>>
{
  public:
    KFMEmptyIdentitySetRemover() = default;
    ;
    ~KFMEmptyIdentitySetRemover() override = default;
    ;


    void ApplyAction(KFMNode<ObjectTypeList>* node) override
    {
        if (node != nullptr) {
            KFMIdentitySet* set = KFMObjectRetriever<ObjectTypeList, KFMIdentitySet>::GetNodeObject(node);

            if (set != nullptr) {
                if (set->GetSize() == 0) {
                    delete set;
                    KFMObjectRetriever<ObjectTypeList, KFMIdentitySet>::SetNodeObject(nullptr, node);
                }
            }
        }
    }

  private:
};

}  // namespace KEMField

#endif /* KFMEmptyIdentitySetRemover_H__ */
