#ifndef KGSubdivisionCondition_HH__
#define KGSubdivisionCondition_HH__

#include "KGCube.hh"
#include "KGIdentitySet.hh"
#include "KGInsertionCondition.hh"
#include "KGInspectingActor.hh"
#include "KGMeshNavigationNode.hh"
#include "KGObjectRetriever.hh"
#include "KGPoint.hh"
#include "KGSpaceTreeProperties.hh"

namespace KGeoBag
{

/*
*
*@file KGSubdivisionCondition.hh
*@class KGSubdivisionCondition
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Aug 26 11:07:01 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KGSubdivisionCondition : public KGInspectingActor<KGMeshNavigationNode>
{
  public:
    KGSubdivisionCondition(unsigned int n_allowed = 1) : fNAllowedElements(n_allowed), fContainer(nullptr){};
    ~KGSubdivisionCondition() override = default;
    ;

    void SetMeshElementContainer(KGNavigableMeshElementContainer* container)
    {
        fContainer = container;
    };

    void SetNAllowedElements(unsigned int n_allowed)
    {
        fNAllowedElements = n_allowed;
    };

    bool ConditionIsSatisfied(KGMeshNavigationNode* node) override;

  private:
    unsigned int fNAllowedElements;

    KGNavigableMeshElementContainer* fContainer;
    KGInsertionCondition fCondition;

    const unsigned int* fDimSize;
    KGPoint<KGMESH_DIM> fLowerCorner;
    KGPoint<KGMESH_DIM> fCenter;
    std::vector<KGCube<KGMESH_DIM>> fCubeScratch;
    double fLength;
};


}  // namespace KGeoBag


#endif /* KGSubdivisionCondition_H__ */
