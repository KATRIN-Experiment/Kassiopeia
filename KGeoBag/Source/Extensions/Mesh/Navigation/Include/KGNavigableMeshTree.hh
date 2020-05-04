#ifndef KGNavigableMeshTree_H__
#define KGNavigableMeshTree_H__

#include "KGMeshNavigationNode.hh"
#include "KGSpaceTree.hh"

namespace KGeoBag
{

/*
*
*@file KGNavigableMeshTree.hh
*@class KGNavigableMeshTree
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Sep 24 15:00:38 EDT 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

//this is the type of tree we operate on
class KGNavigableMeshTree : public KGSpaceTree<KGMESH_DIM, KGMeshNavigationNodeObjects>
{
  public:
    KGNavigableMeshTree() : KGSpaceTree<KGMESH_DIM, KGMeshNavigationNodeObjects>()
    {
        ;
    }
    ~KGNavigableMeshTree() override
    {
        //set the element container pointer to null for all nodes so we dont delete it more than once
        KGNodeObjectNullifier<KGMeshNavigationNodeObjects, KGNavigableMeshElementContainer> containerNullifier;
        this->ApplyCorecursiveAction(&containerNullifier);
    };
};


}  // namespace KGeoBag


#endif /* end of include guard: KGNavigableMeshTree_H__ */
