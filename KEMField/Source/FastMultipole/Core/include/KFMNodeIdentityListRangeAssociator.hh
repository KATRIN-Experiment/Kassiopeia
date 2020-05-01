#ifndef KFMNodeIdentityListRangeAssociator_HH__
#define KFMNodeIdentityListRangeAssociator_HH__


namespace KEMField
{

/*
*
*@file KFMNodeIdentityListRangeAssociator.hh
*@class KFMNodeIdentityListRangeAssociator
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sat Jun 14 20:14:04 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ObjectTypeList>
class KFMNodeIdentityListRangeAssociator : public KFMNodeActor<KFMNode<ObjectTypeList>>
{
  public:
    KFMNodeIdentityListRangeAssociator(){};
    virtual ~KFMNodeIdentityListRangeAssociator(){};

    void Clear()
    {
        fNodeStartIndexList.clear();
        fNodeIndexListSize.clear();
    }

    const std::vector<unsigned int>* GetNodeStartIndexList() const
    {
        return &fNodeStartIndexList;
    };
    void GetNodeStartIndexList(std::vector<unsigned int>* list) const
    {
        *list = fNodeStartIndexList;
    };

    const std::vector<unsigned int>* GetNodeIndexListSize() const
    {
        return &fNodeIndexListSize;
    };
    void GetNodeIndexListSize(std::vector<unsigned int>* list) const
    {
        *list = fNodeIndexListSize;
    };

    virtual void ApplyAction(KFMNode<ObjectTypeList>* node)
    {
        if (node != NULL) {
            unsigned int id = node->GetID();

            KFMNodeIdentityListRange* id_range = nullptr;
            id_range = KFMObjectRetriever<ObjectTypeList, KFMNodeIdentityListRange>::GetNodeObject(node);

            if (fNodeIndexListSize.size() <= id) {
                fNodeIndexListSize.resize(id + 1);
                fNodeStartIndexList.resize(id + 1);
            }

            if (id_range != nullptr) {
                unsigned int start_index = id_range->GetStartIndex();
                unsigned int size = id_range->GetLength();
                fNodeStartIndexList[id] = start_index;
                fNodeIndexListSize[id] = size;
            }
            else {
                fNodeStartIndexList[id] = 0;
                fNodeIndexListSize[id] = 0;
            }
        }
    }

  private:
    //these lists are indexed by node id
    std::vector<unsigned int> fNodeStartIndexList;
    std::vector<unsigned int> fNodeIndexListSize;
};


}  // namespace KEMField


#endif /* KFMNodeIdentityListRangeAssociator_H__ */
