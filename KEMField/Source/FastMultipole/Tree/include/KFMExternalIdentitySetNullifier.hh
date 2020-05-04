#ifndef KFMExternalIdentitySetNullifier_HH__
#define KFMExternalIdentitySetNullifier_HH__


namespace KEMField
{

/*
*
*@file KFMExternalIdentitySetNullifier.hh
*@class KFMExternalIdentitySetNullifier
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sun May 25 10:58:51 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<typename ObjectTypeList> class KFMExternalIdentitySetNullifier : public KFMNodeActor<KFMNode<ObjectTypeList>>
{
  public:
    KFMExternalIdentitySetNullifier()
    {
        ;
    };
    virtual ~KFMExternalIdentitySetNullifier(){};

    virtual void ApplyAction(KFMNode<ObjectTypeList>* node)
    {
        if (node != NULL) {
            if (node->GetParent() != NULL) {
                KFMExternalIdentitySet* parent_eid_set =
                    KFMObjectRetriever<ObjectTypeList, KFMExternalIdentitySet>::GetNodeObject(node->GetParent());
                KFMExternalIdentitySet* eid_set =
                    KFMObjectRetriever<ObjectTypeList, KFMExternalIdentitySet>::GetNodeObject(node);

                if (eid_set == parent_eid_set) {
                    //null out the childs pointer to the parents id set
                    KFMObjectRetriever<ObjectTypeList, KFMExternalIdentitySet>::SetNodeObject(NULL, node);
                }
            }
        }
    }

  private:
};


}  // namespace KEMField


#endif /* KFMExternalIdentitySetNullifier_H__ */
