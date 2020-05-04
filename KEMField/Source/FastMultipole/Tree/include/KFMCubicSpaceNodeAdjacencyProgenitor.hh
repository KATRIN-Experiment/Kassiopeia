#ifndef KFMCubicSpaceNodeAdjacencyProgenitor_HH__
#define KFMCubicSpaceNodeAdjacencyProgenitor_HH__

#include "KFMCubicSpaceNodeNeighborFinder.hh"
#include "KFMCubicSpaceNodeProgenitor.hh"
#include "KFMInspectingActor.hh"
#include "KFMNodeFlags.hh"

#include <cmath>
#include <cstdlib>

namespace KEMField
{

/*
*
*@file KFMCubicSpaceNodeAdjacencyProgenitor.hh
*@class KFMCubicSpaceNodeAdjacencyProgenitor
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Jan 27 12:52:03 EST 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ObjectTypeList, unsigned int SpatialNDIM>
class KFMCubicSpaceNodeAdjacencyProgenitor : public KFMNodeActor<KFMNode<ObjectTypeList>>
{
  public:
    KFMCubicSpaceNodeAdjacencyProgenitor()
    {
        fProgenitor = new KFMCubicSpaceNodeProgenitor<SpatialNDIM, ObjectTypeList>();
        fZeroMaskSize = 0;
    };

    ~KFMCubicSpaceNodeAdjacencyProgenitor() override
    {
        delete fProgenitor;
    };

    virtual void SetZeroMaskSize(int zmask)
    {
        fZeroMaskSize = std::abs(zmask);
    };

    void ApplyAction(KFMNode<ObjectTypeList>* node) override
    {
        if (node != nullptr) {
            //we expect that some external condition has been satisfied
            //now visit all this nodes neighbors
            //and if they do not have any children present
            //we given them children
            KFMCubicSpaceNodeNeighborFinder<SpatialNDIM, ObjectTypeList>::GetAllNeighbors(node,
                                                                                          fZeroMaskSize,
                                                                                          &fNeighborNodeList);

            for (unsigned int i = 0; i < fNeighborNodeList.size(); i++) {
                if (fNeighborNodeList[i] != nullptr) {
                    if (!(fNeighborNodeList[i]->HasChildren())) {
                        fProgenitor->ApplyAction(fNeighborNodeList[i]);
                    }
                }
            }
        }
    }


  private:
    int fZeroMaskSize;
    KFMCubicSpaceNodeProgenitor<SpatialNDIM, ObjectTypeList>* fProgenitor;
    std::vector<KFMNode<ObjectTypeList>*> fNeighborNodeList;
};

}  // namespace KEMField


#endif /* KFMCubicSpaceNodeAdjacencyProgenitor_H__ */
