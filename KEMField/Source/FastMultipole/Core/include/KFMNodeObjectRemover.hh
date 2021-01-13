#ifndef KFMNodeObjectRemover_HH__
#define KFMNodeObjectRemover_HH__

#include "KFMNode.hh"
#include "KFMNodeActor.hh"
#include "KFMObjectRetriever.hh"


namespace KEMField
{


/*
*
*@file KFMNodeObjectRemover.hh
*@class KFMNodeObjectRemover
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Sep 24 17:14:40 EDT 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ObjectTypeList, typename TypeToRemove>
class KFMNodeObjectRemover : public KFMNodeActor<KFMNode<ObjectTypeList>>
{
  public:
    KFMNodeObjectRemover() = default;
    ;
    ~KFMNodeObjectRemover() override = default;
    ;

    void ApplyAction(KFMNode<ObjectTypeList>* node) override
    {
        TypeToRemove* remove_this_object = KFMObjectRetriever<ObjectTypeList, TypeToRemove>::GetNodeObject(node);

        if (remove_this_object != nullptr) {
            delete remove_this_object;
            KFMObjectRetriever<ObjectTypeList, TypeToRemove>::SetNodeObject(nullptr, node);
        }
    }

  private:
};


}  // namespace KEMField

#endif /* KFMNodeObjectRemover_H__ */
