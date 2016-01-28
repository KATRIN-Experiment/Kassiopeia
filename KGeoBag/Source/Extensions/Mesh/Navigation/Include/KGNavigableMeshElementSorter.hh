#ifndef KGNavigableMeshElementSorter_HH__
#define KGNavigableMeshElementSorter_HH__

#include <cmath>

#include "KGMeshNavigationNode.hh"

#include "KGNodeActor.hh"
#include "KGObjectRetriever.hh"
#include "KGObjectContainer.hh"
#include "KGIdentitySet.hh"

#include "KGNavigableMeshElement.hh"
#include "KGInsertionCondition.hh"

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

class KGNavigableMeshElementSorter: public KGNodeActor< KGMeshNavigationNode >
{
    public:
        KGNavigableMeshElementSorter():fContainer(NULL){};
        virtual ~KGNavigableMeshElementSorter(){};

        void SetMeshElementContainer(KGNavigableMeshElementContainer* container){fContainer = container;};

        virtual void ApplyAction( KGMeshNavigationNode* node);

    private:

        KGNavigableMeshElementContainer* fContainer;
        KGInsertionCondition fCondition;

};


}

#endif /* KGNavigableMeshElementSorter_H__ */
