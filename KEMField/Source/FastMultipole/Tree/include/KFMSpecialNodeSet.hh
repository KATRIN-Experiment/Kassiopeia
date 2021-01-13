#ifndef __KFMSpecialNodeSet_H__
#define __KFMSpecialNodeSet_H__

#include "KFMMessaging.hh"
#include "KFMNode.hh"

#include <vector>


namespace KEMField
{

/**
*
*@file KFMSpecialNodeSet.hh
*@class KFMSpecialNodeSet
*@brief bidirectional map of various node ids (and their pointers) that satisfy some special condition
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Sun Jul  6 13:50:26 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ObjectTypeList> class KFMSpecialNodeSet
{
  public:
    KFMSpecialNodeSet()
    {
        fNTotalNodes = 0;
        fSize = 0;
    };

    virtual ~KFMSpecialNodeSet() = default;
    ;

    void SetTotalNumberOfNodes(unsigned int n_nodes)
    {
        fNTotalNodes = n_nodes;
        fOrdinaryToSpecial.clear();
        fOrdinaryToSpecial.resize(fNTotalNodes, -1);
        fSpecialToOrdinary.clear();
        fSpecialNodes.clear();
        fSize = 0;
    };


    void AddNode(KFMNode<ObjectTypeList>* node)
    {
        unsigned int ordinary_id = node->GetID();
        //first we check if the node is already present in our special set
        int test_id = GetSpecializedIDFromOrdinaryID(ordinary_id);

        //if node is not yet present, add it to the set,
        //otherwise, since we already have it, do nothing
        if (test_id == -1) {
            fOrdinaryToSpecial[ordinary_id] = fSize;
            fSpecialToOrdinary.push_back(ordinary_id);
            fSpecialNodes.push_back(node);
            fSize++;
        }
    }

    unsigned int GetSize() const
    {
        return fSize;
    };

    //returns -1 if node not present in special set
    int GetOrdinaryIDFromSpecializedID(unsigned int special_id)
    {
        if (special_id < fSize) {
            return fSpecialToOrdinary[special_id];
        }
        else {
            return -1;
        }
    }

    //returns -1 if node not present in special set
    int GetSpecializedIDFromOrdinaryID(unsigned int ordinary_id)
    {
        if (ordinary_id < fNTotalNodes) {
            return fOrdinaryToSpecial[ordinary_id];
        }
        else {
            kfmout
                << "KFMSpecialNodeSet::GetSpecializedIDFromOrdinaryID: Error, node id exceeds the total number of nodes."
                << kfmendl;
            kfmexit(1);
            return -1;
        }
    }

    //returns NULL if node not present in special set
    KFMNode<ObjectTypeList>* GetNodeFromSpecializedID(unsigned int special_id)
    {
        if (special_id < fSize) {
            return fSpecialNodes[special_id];
        }
        else {
            return NULL;
        }
    }

    //returns NULL if node not present in special set
    KFMNode<ObjectTypeList>* GetNodeFromOrdinaryID(unsigned int ordinary_id)
    {
        int special_id = GetSpecializedIDFromOrdinaryID(ordinary_id);
        if (special_id == -1) {
            return NULL;
        };
        return GetNodeFromSpecializedID(special_id);
    }

  protected:
    /* data */

    unsigned int fSize;
    unsigned int fNTotalNodes;
    std::vector<int> fOrdinaryToSpecial;
    std::vector<int> fSpecialToOrdinary;
    std::vector<KFMNode<ObjectTypeList>*> fSpecialNodes;
};


}  // namespace KEMField

#endif /* __KFMSpecialNodeSet_H__ */
