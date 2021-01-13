#ifndef KFMNodeObjectNullifier_HH__
#define KFMNodeObjectNullifier_HH__

#include "KFMNode.hh"
#include "KFMNodeActor.hh"
#include "KFMObjectRetriever.hh"


namespace KEMField
{


/*
*
*@file KFMNodeObjectNullifier.hh
*@class KFMNodeObjectNullifier
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Sep 24 17:14:40 EDT 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ObjectTypeList, typename TypeToRemove>
class KFMNodeObjectNullifier : public KFMNodeActor<KFMNode<ObjectTypeList>>
{
  public:
    KFMNodeObjectNullifier() = default;
    ;
    ~KFMNodeObjectNullifier() override = default;
    ;

    void ApplyAction(KFMNode<ObjectTypeList>* node) override
    {
        //does not delete the object, just sets the pointer to null, this is useful
        //when many nodes point to the same object, which has just been deleted
        KFMObjectRetriever<ObjectTypeList, TypeToRemove>::SetNodeObject(nullptr, node);
    }

  private:
};


}  // namespace KEMField

#endif /* KFMNodeObjectNullifier_H__ */
