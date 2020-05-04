#ifndef KFMTreeStructureExtractor_HH__
#define KFMTreeStructureExtractor_HH__

#include "KFMNode.hh"
#include "KFMNodeActor.hh"

namespace KEMField
{

/*
*
*@file KFMTreeStructureExtractor.hh
*@class KFMTreeStructureExtractor
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Apr  2 14:17:04 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<typename ObjectTypeList> class KFMTreeStructureExtractor : public KFMNodeActor<KFMNode<ObjectTypeList>>
{
  public:
    KFMTreeStructureExtractor()
    {
        fFlattenedTree.clear();
        fNNodes = 0;
    };
    ~KFMTreeStructureExtractor() override{};

    void ApplyAction(KFMNode<ObjectTypeList>* node) override
    {
        if (node != nullptr) {
            //we only need to actively keep track of non-leaf nodes
            if (node->HasChildren()) {
                unsigned int id = node->GetID();
                std::vector<unsigned int> child_ids;
                child_ids.resize(0);

                unsigned int n_children = node->GetNChildren();
                child_ids.reserve(n_children);
                for (unsigned int i = 0; i < n_children; i++) {
                    child_ids.push_back(node->GetChild(i)->GetID());
                }

                KFMNodeData data;
                data.SetID(id);
                data.SetChildIDs(&child_ids);

                fFlattenedTree.push_back(data);
            }

            fNNodes++;
        }
    }

    unsigned int GetNumberOfNodes() const
    {
        return fNNodes;
    };

    const std::vector<KFMNodeData>* GetFlattenedTree() const
    {
        return &fFlattenedTree;
    };


  private:
    unsigned int fNNodes;
    std::vector<KFMNodeData> fFlattenedTree;
};


}  // namespace KEMField

#endif /* KFMTreeStructureExtractor_H__ */
