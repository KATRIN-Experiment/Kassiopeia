#ifndef KGNavigableMeshElementSorter_HH__
#define KGNavigableMeshElementSorter_HH__

#include "KGIdentitySet.hh"
#include "KGInsertionCondition.hh"
#include "KGMeshNavigationNode.hh"
#include "KGNavigableMeshElement.hh"
#include "KGNodeActor.hh"
#include "KGObjectContainer.hh"
#include "KGObjectRetriever.hh"

#include <cmath>

namespace KGeoBag
{

/*
*
*@file KGNavigableMeshElementSorter.hh
*@class KGNavigableMeshElementSorter
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sun Aug 25 19:32:36 CEST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KGNavigableMeshElementSorter : public KGNodeActor<KGMeshNavigationNode>
{
  public:
    KGNavigableMeshElementSorter() : fContainer(nullptr){};
    ~KGNavigableMeshElementSorter() override{};

    void SetMeshElementContainer(KGNavigableMeshElementContainer* container)
    {
        fContainer = container;
    };

    void ApplyAction(KGMeshNavigationNode* node) override;

  private:
    KGNavigableMeshElementContainer* fContainer;
    KGInsertionCondition fCondition;
};


}  // namespace KGeoBag

#endif /* KGNavigableMeshElementSorter_H__ */
