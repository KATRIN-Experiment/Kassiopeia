#ifndef KFMNodeIdentityListCreator_HH__
#define KFMNodeIdentityListCreator_HH__

#include "KFMIdentitySet.hh"
#include "KFMNode.hh"
#include "KFMNodeActor.hh"
#include "KFMNodeIdentityListRange.hh"

#include <vector>

namespace KEMField
{

/*
*
*@file KFMNodeIdentityListCreator.hh
*@class KFMNodeIdentityListCreator
*@brief This visitor MUST be applied recursively with REVERSE visiting order
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sat Jun 14 13:41:32 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ObjectTypeList> class KFMNodeIdentityListCreator : public KFMNodeActor<KFMNode<ObjectTypeList>>
{
  public:
    KFMNodeIdentityListCreator(){};
    virtual ~KFMNodeIdentityListCreator(){};

    void Clear()
    {
        fIdentityList.clear();
    }

    const std::vector<unsigned int>* GetIdentityList() const
    {
        return &fIdentityList;
    };
    void GetIdentityList(std::vector<unsigned int>* list) const
    {
        *list = fIdentityList;
    };

    virtual void ApplyAction(KFMNode<ObjectTypeList>* node)
    {
        if (node != NULL) {
            KFMIdentitySet* id_set = nullptr;
            id_set = KFMObjectRetriever<ObjectTypeList, KFMIdentitySet>::GetNodeObject(node);

            KFMNodeIdentityListRange* id_range = nullptr;
            id_range = KFMObjectRetriever<ObjectTypeList, KFMNodeIdentityListRange>::GetNodeObject(node);

            if (id_range == nullptr) {
                id_range = new KFMNodeIdentityListRange();
                KFMObjectRetriever<ObjectTypeList, KFMNodeIdentityListRange>::SetNodeObject(id_range, node);
            }
            else {
                delete id_range;
                id_range = nullptr;
                id_range = new KFMNodeIdentityListRange();
                KFMObjectRetriever<ObjectTypeList, KFMNodeIdentityListRange>::SetNodeObject(id_range, node);
            }

            unsigned int start_index = fIdentityList.size();
            unsigned int size = 0;

            if (node->HasChildren()) {
                //loop over children looking for the earliest start index
                for (unsigned int i = 0; i < node->GetNChildren(); i++) {
                    KFMNode<ObjectTypeList>* child = node->GetChild(i);
                    KFMNodeIdentityListRange* child_id_range = nullptr;
                    child_id_range = KFMObjectRetriever<ObjectTypeList, KFMNodeIdentityListRange>::GetNodeObject(child);

                    if (child_id_range != nullptr) {
                        if (child_id_range->GetStartIndex() < start_index) {
                            start_index = child_id_range->GetStartIndex();
                        }
                    }
                }

                size = fIdentityList.size() - start_index;
            }

            //whether or not node has children, we need to add elements that
            //it might own itself
            if (id_set != nullptr) {
                size += id_set->GetSize();
                //push the ids onto the list
                std::vector<unsigned int> ids;
                id_set->GetIDs(&ids);
                for (unsigned int i = 0; i < ids.size(); i++) {
                    fIdentityList.push_back(ids[i]);
                }
            }

            //set up this nodes list range
            id_range->SetStartIndex(start_index);
            id_range->SetLength(size);
        }
    }

  private:
    std::vector<unsigned int> fIdentityList;
};


}  // namespace KEMField

#endif /* KFMNodeIdentityListCreator_H__ */
