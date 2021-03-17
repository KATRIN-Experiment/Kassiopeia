#ifndef KGNodeObjectNullifier_HH__
#define KGNodeObjectNullifier_HH__

#include "KGNode.hh"
#include "KGNodeActor.hh"
#include "KGObjectRetriever.hh"


namespace KGeoBag
{


/*
*
*@file KGNodeObjectNullifier.hh
*@class KGNodeObjectNullifier
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Sep 24 17:14:40 EDT 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ObjectTypeList, typename TypeToRemove>
class KGNodeObjectNullifier : public KGNodeActor<KGNode<ObjectTypeList>>
{
  public:
    KGNodeObjectNullifier() = default;
    ;
    ~KGNodeObjectNullifier() override = default;
    ;

    void ApplyAction(KGNode<ObjectTypeList>* node) override
    {
        //does not delete the object, just sets the pointer to null, this is useful
        //when many nodes point to the same object, which has just been deleted
        KGObjectRetriever<ObjectTypeList, TypeToRemove>::SetNodeObject(nullptr, node);
    }

  private:
};


}  // namespace KGeoBag

#endif /* KGNodeObjectNullifier_H__ */
