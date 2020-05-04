#ifndef KFMObjectCollector_HH__
#define KFMObjectCollector_HH__


namespace KEMField
{

/*
*
*@file KFMObjectCollector.hh
*@class KFMObjectCollector
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Apr  3 10:01:40 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ObjectTypeList, typename CollectedObjectType>
class KFMObjectCollector : public KFMNodeActor<KFMNode<ObjectTypeList>>
{
  public:
    KFMObjectCollector(){};
    ~KFMObjectCollector() override{};

    void Clear()
    {
        fCollectedObjects.clear();
        fNodeIDs.clear();
    };

    const std::vector<CollectedObjectType*>* GetCollectedObjects() const
    {
        return &fCollectedObjects;
    };
    const std::vector<int>* GetCollectedObjectAssociatedNodeIDs() const
    {
        return &fNodeIDs;
    };

    void ApplyAction(KFMNode<ObjectTypeList>* node) override
    {
        if (node != nullptr) {
            int id = node->GetID();
            CollectedObjectType* obj = KFMObjectRetriever<ObjectTypeList, CollectedObjectType>::GetNodeObject(node);

            if (obj != nullptr) {
                fCollectedObjects.push_back(obj);
                fNodeIDs.push_back(id);
            }
        }
    }

  private:
    std::vector<int> fNodeIDs;
    std::vector<CollectedObjectType*> fCollectedObjects;
};


}  // namespace KEMField

#endif /* KFMObjectCollector_H__ */
