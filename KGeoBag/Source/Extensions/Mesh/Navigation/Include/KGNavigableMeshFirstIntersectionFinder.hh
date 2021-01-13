#ifndef KGNavigableMeshFirstIntersectionFinder_H__
#define KGNavigableMeshFirstIntersectionFinder_H__

#include "KGArrayMath.hh"
#include "KGBoundaryCalculator.hh"
#include "KGCube.hh"
#include "KGIdentitySet.hh"
#include "KGMeshNavigationNode.hh"
#include "KGNavigableMeshElement.hh"
#include "KGNavigableMeshElementContainer.hh"
#include "KGNavigableMeshTree.hh"
#include "KGPoint.hh"

#include <cmath>
#include <complex>
#include <cstdlib>
#include <stack>
#include <utility>
#include <vector>


namespace KGeoBag
{

/**
*
*@file KGNavigableMeshFirstIntersectionFinder.hh
*@class KGNavigableMeshFirstIntersectionFinder
*@brief finds the first intersection (closest to start point) of a line segment and a mesh
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Jul 17 11:29:45 EDT 2015 J. Barrett (barrettj@mit.edu) First Version
*
*/


class KGNavigableMeshFirstIntersectionFinder : public KGNodeActor<KGMeshNavigationNode>
{
  public:
    KGNavigableMeshFirstIntersectionFinder();
    ~KGNavigableMeshFirstIntersectionFinder() override;

    void SetMeshElementContainer(KGNavigableMeshElementContainer* container)
    {
        fContainer = container;
    };
    void SetSilent()
    {
        fVerbose = false;
    };
    void SetVerbose()
    {
        fVerbose = true;
    };

    void NearestPointOnLineSegment(const KGeoBag::KThreeVector& aPoint, KGeoBag::KThreeVector& aNearest,
                                   double& t) const;
    double LineSegmentDistanceToPoint(const KGeoBag::KThreeVector& aPoint) const;
    bool LineSegmentIntersectsCube(const KGCube<KGMESH_DIM>& cube, double& distance) const;


    void SetLineSegment(const KGeoBag::KThreeVector& start, const KGeoBag::KThreeVector& end);
    bool HasIntersectionWithMesh() const;
    KGeoBag::KThreeVector GetIntersection() const;
    const KGNavigableMeshElement* GetIntersectedMeshElement() const
    {
        return fIntersectedElement;
    };

    void ApplyAction(KGMeshNavigationNode* node) override;

  private:
    bool fVerbose;

    //mesh element container
    KGNavigableMeshElementContainer* fContainer;

    struct ChildDistanceOrder
    {
        bool operator()(std::pair<KGMeshNavigationNode*, double> a, std::pair<KGMeshNavigationNode*, double> b)
        {
            return (a.second > b.second);  //this will sort them from farthest to nearest
        }
    };


    //stack space and functions for tree traversal
    unsigned int fDefaultStackSize;
    unsigned int fStackReallocateLimit;
    typedef KGMeshNavigationNode* KGMeshNavigationNodePtr;
    KGMeshNavigationNodePtr* fPreallocatedStackTopPtr;
    std::vector<KGMeshNavigationNode*> fPreallocatedStack;
    unsigned int fStackSize;

    //sort the intersected child node by distance from line segment start
    static void SortOctreeNodes(unsigned int n_nodes, std::pair<KGMeshNavigationNode*, double>* nodes);

    inline void CheckStackSize()
    {
        if (fStackSize >= fStackReallocateLimit) {
            fPreallocatedStack.resize(3 * fStackSize);
            fStackReallocateLimit = 2 * fStackSize;
        }
    };

    //parameters of the line segment
    KGeoBag::KThreeVector fStartPoint;
    KGeoBag::KThreeVector fEndPoint;
    KGeoBag::KThreeVector fDirection;
    double fLength;

    KGMeshNavigationNode* fTempNode;
    std::pair<KGMeshNavigationNode*, double> fOrderedChildren[8];

    //intersection data
    bool fHaveIntersection;
    KGeoBag::KThreeVector fFirstIntersection;
    KGNavigableMeshElement* fIntersectedElement;
};


}  // namespace KGeoBag

#endif /* end of include guard: KGNavigableMeshFirstIntersectionFinder_H__ */
