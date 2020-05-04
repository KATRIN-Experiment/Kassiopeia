#ifndef __KFMElementInfluenceRangeCollector_H__
#define __KFMElementInfluenceRangeCollector_H__

#include "KFMIdentitySet.hh"
#include "KFMNode.hh"
#include "KFMNodeActor.hh"
#include "KFMNodeIdentityListRange.hh"

namespace KEMField
{

/**
*
*@file KFMElementInfluenceRangeCollector.hh
*@class KFMElementInfluenceRangeCollector
*@brief
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Aug 27 22:26:05 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<unsigned int NDIM, typename ObjectTypeList>
class KFMElementInfluenceRangeCollector : public KFMNodeActor<KFMNode<ObjectTypeList>>
{
  public:
    KFMElementInfluenceRangeCollector() : fZeroMaskSize(0){};
    virtual ~KFMElementInfluenceRangeCollector(){};

    void SetZeroMaskSize(unsigned int zmask)
    {
        fZeroMaskSize = zmask;
    };

    virtual void ApplyAction(KFMNode<ObjectTypeList>* node)
    {
        //This actor must visit the tree after the
        //KFMNodeIdentityListCreator or nothing will happen (the list/ranges won't yet exist)

        if (node != NULL) {
            //retrieve this nodes KFMElementLocalInfluenceRange and delete it if it already exists
            KFMElementLocalInfluenceRange* influence_range = nullptr;
            influence_range = KFMObjectRetriever<ObjectTypeList, KFMElementLocalInfluenceRange>::GetNodeObject(node);
            if (influence_range != nullptr) {
                delete influence_range;
                influence_range = nullptr;
            };
            KFMObjectRetriever<ObjectTypeList, KFMElementLocalInfluenceRange>::SetNodeObject(NULL, node);

            //node must have a non-empty id-set
            KFMIdentitySet* id_set = nullptr;
            id_set = KFMObjectRetriever<ObjectTypeList, KFMIdentitySet>::GetNodeObject(node);

            if (id_set != nullptr) {
                if (id_set->GetSize() != 0) {
                    fNodeNeighborList.clear();
                    KFMCubicSpaceNodeNeighborFinder<NDIM, ObjectTypeList>::GetAllNeighbors(node,
                                                                                           fZeroMaskSize,
                                                                                           &fNodeNeighborList);

                    influence_range = new KFMElementLocalInfluenceRange();

                    for (unsigned int j = 0; j < fNodeNeighborList.size(); j++) {
                        if (fNodeNeighborList[j] != NULL) {
                            KFMNodeIdentityListRange* range = nullptr;
                            range = KFMObjectRetriever<ObjectTypeList, KFMNodeIdentityListRange>::GetNodeObject(
                                fNodeNeighborList[j]);

                            if (range != nullptr) {
                                if (range->GetLength() != 0) {
                                    //we add this range to the current node's range list
                                    influence_range->AddRange(range->GetStartIndex(), range->GetLength());
                                }
                            }
                        }
                    }

                    if (influence_range->IsEmpty()) {
                        delete influence_range;
                        KFMObjectRetriever<ObjectTypeList, KFMElementLocalInfluenceRange>::SetNodeObject(NULL, node);
                    }
                    else {
                        KFMObjectRetriever<ObjectTypeList, KFMElementLocalInfluenceRange>::SetNodeObject(
                            influence_range,
                            node);
                    }
                }
            }
        }
    }

  private:
    unsigned int fZeroMaskSize;
    std::vector<KFMNode<ObjectTypeList>*> fNodeNeighborList;
};


}  // namespace KEMField

#endif /* __KFMElementInfluenceRangeCollector_H__ */
