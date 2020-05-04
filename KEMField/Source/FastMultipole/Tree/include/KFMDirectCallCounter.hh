#ifndef KFMDirectCallCounter_HH__
#define KFMDirectCallCounter_HH__

#include "KFMIdentitySetList.hh"
#include "KFMNode.hh"
#include "KFMNodeActor.hh"
#include "KFMObjectRetriever.hh"

#include <vector>

namespace KEMField
{

/*
*
*@file KFMDirectCallCounter.hh
*@class KFMDirectCallCounter
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Jan 23 16:59:56 EST 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ObjectTypeList> class KFMDirectCallCounter : public KFMNodeActor<KFMNode<ObjectTypeList>>
{
  public:
    KFMDirectCallCounter()
    {
        fMaxDirectCalls = 0;
    };
    ~KFMDirectCallCounter() override{};

    unsigned int GetMaxDirectCalls() const
    {
        return fMaxDirectCalls;
    };

    void ApplyAction(KFMNode<ObjectTypeList>* node) override
    {
        if (node != nullptr) {
            if (!(node->HasChildren()))  //only apply to leaf nodes
            {
                fNodeList.clear();
                //collect all of the parent nodes of this node
                KFMNode<ObjectTypeList>* temp_node = node;
                do {
                    if (temp_node != nullptr) {
                        fNodeList.push_back(temp_node);
                        temp_node = temp_node->GetParent();
                    }
                } while (temp_node != nullptr);

                //loop over the node list and collect the direct call elements from their id set lists
                unsigned int subset_size = 0;
                unsigned int n_nodes = fNodeList.size();
                for (unsigned int i = 0; i < n_nodes; i++) {
                    if (fNodeList[i] != nullptr) {
                        KFMIdentitySetList* id_set_list =
                            KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMIdentitySetList>::GetNodeObject(
                                fNodeList[i]);

                        if (id_set_list != nullptr) {
                            unsigned int n_sets = id_set_list->GetNumberOfSets();
                            for (unsigned int j = 0; j < n_sets; j++) {
                                subset_size += id_set_list->GetSet(j)->size();
                            }
                        }
                    }
                }

                if (fMaxDirectCalls < subset_size) {
                    fMaxDirectCalls = subset_size;
                }
            }
        }
    }

  private:
    unsigned int fMaxDirectCalls;
    std::vector<KFMNode<ObjectTypeList>*> fNodeList;
};


}  // namespace KEMField

#endif /* KFMDirectCallCounter_H__ */
