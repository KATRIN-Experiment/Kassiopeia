#ifndef KGObjectCollector_HH__
#define KGObjectCollector_HH__


namespace KGeoBag
{

/*
*
*@file KGObjectCollector.hh
*@class KGObjectCollector
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Apr  3 10:01:40 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ObjectTypeList, typename CollectedObjectType>
class KGObjectCollector : public KGNodeActor<KGNode<ObjectTypeList>>
{
  public:
    KGObjectCollector(){};
    ~KGObjectCollector() override{};

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

    void ApplyAction(KGNode<ObjectTypeList>* node) override
    {
        if (node != nullptr) {
            int id = node->GetID();
            CollectedObjectType* obj = KGObjectRetriever<ObjectTypeList, CollectedObjectType>::GetNodeObject(node);

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


}  // namespace KGeoBag

#endif /* KGObjectCollector_H__ */
