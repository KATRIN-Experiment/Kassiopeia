#ifndef KFMNodeCollector_HH__
#define KFMNodeCollector_HH__


#include "KFMNode.hh"
#include "KFMNodeActor.hh"

#include <map>
#include <vector>

namespace KEMField
{

/**
*
*@file KFMNodeCollector.hh
*@class KFMNodeCollector
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sun Jun 29 15:37:47 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ObjectTypeList> class KFMNodeCollector : public KFMNodeActor<KFMNode<ObjectTypeList>>
{
  public:
    KFMNodeCollector(){};
    virtual ~KFMNodeCollector(){};

    void SetListOfNodeIDs(std::vector<unsigned int>* node_ids)
    {
        fNodeIDs = *node_ids;
        fNodes.clear();
        fNodes.resize(fNodeIDs.size(), NULL);

        fNodeID2IndexMap.clear();
        for (unsigned int i = 0; i < fNodeIDs.size(); i++) {
            fNodeID2IndexMap.insert(std::pair<unsigned int, unsigned int>(fNodeIDs[i], i));
        }
    }

    void GetNodeList(std::vector<KFMNode<ObjectTypeList>*>* nodes)
    {
        *nodes = fNodes;
    }

    virtual void ApplyAction(KFMNode<ObjectTypeList>* node)
    {
        if (node != NULL) {
            unsigned int id = node->GetID();

            int collection_index = GetCollectionIndex(id);
            if (collection_index != -1) {
                fNodes[collection_index] = node;
            }
        }
    }

  protected:
    int GetCollectionIndex(unsigned int id)
    {
        std::map<unsigned int, unsigned int>::iterator it = fNodeID2IndexMap.find(id);

        if (it != fNodeID2IndexMap.end()) {
            return (int) (it->second);
        }
        else {
            return -1;
        }
    }

    //list of the ids of the nodes we are suppose to collect
    std::vector<unsigned int> fNodeIDs;

    //list of the pointers to the collected nodes
    std::vector<KFMNode<ObjectTypeList>*> fNodes;

    std::map<unsigned int, unsigned int> fNodeID2IndexMap;
};

}  // namespace KEMField

#endif /* KFMNodeCollector_H__ */
