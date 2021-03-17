#ifndef KGEmptyIdentitySetRemover_HH__
#define KGEmptyIdentitySetRemover_HH__

//#include "KGExternalIdentitySet.hh"
#include "KGIdentitySet.hh"
#include "KGNode.hh"
#include "KGNodeActor.hh"
#include "KGObjectRetriever.hh"

namespace KGeoBag
{

/*
*
*@file KGEmptyIdentitySetRemover.hh
*@class KGEmptyIdentitySetRemover
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Apr  3 12:02:46 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ObjectTypeList> class KGEmptyIdentitySetRemover : public KGNodeActor<KGNode<ObjectTypeList>>
{
  public:
    KGEmptyIdentitySetRemover() = default;
    ;
    virtual ~KGEmptyIdentitySetRemover() = default;
    ;


    virtual void ApplyAction(KGNode<ObjectTypeList>* node)
    {
        if (node != NULL) {
            KGIdentitySet* set = KGObjectRetriever<ObjectTypeList, KGIdentitySet>::GetNodeObject(node);

            if (set != nullptr) {
                if (set->GetSize() == 0) {
                    delete set;
                    KGObjectRetriever<ObjectTypeList, KGIdentitySet>::SetNodeObject(NULL, node);
                }
            }

            /*
            KGExternalIdentitySet* eset = KGObjectRetriever<ObjectTypeList, KGExternalIdentitySet>::GetNodeObject(node);

            if (eset != NULL) {
                if (eset->GetSize() == 0) {
                    delete eset;
                    KGObjectRetriever<ObjectTypeList, KGExternalIdentitySet>::SetNodeObject(NULL, node);
                }
            }
            */
        }
    }

  private:
};

}  // namespace KGeoBag

#endif /* KGEmptyIdentitySetRemover_H__ */
