#ifndef KGTreeStructureExtractor_HH__
#define KGTreeStructureExtractor_HH__

#include "KGNode.hh"
#include "KGNodeActor.hh"
#include "KGNodeData.hh"

namespace KGeoBag
{

/*
*
*@file KGTreeStructureExtractor.hh
*@class KGTreeStructureExtractor
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Apr  2 14:17:04 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<typename ObjectTypeList> class KGTreeStructureExtractor : public KGNodeActor<KGNode<ObjectTypeList>>
{
  public:
    KGTreeStructureExtractor()
    {
        fFlattenedTree.clear();
        fNNodes = 0;
    };
    ~KGTreeStructureExtractor() override{};

    void ApplyAction(KGNode<ObjectTypeList>* node) override
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

                KGNodeData data;
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

    const std::vector<KGNodeData>* GetFlattenedTree() const
    {
        return &fFlattenedTree;
    };


  private:
    unsigned int fNNodes;
    std::vector<KGNodeData> fFlattenedTree;
};


}  // namespace KGeoBag

#endif /* KGTreeStructureExtractor_H__ */
