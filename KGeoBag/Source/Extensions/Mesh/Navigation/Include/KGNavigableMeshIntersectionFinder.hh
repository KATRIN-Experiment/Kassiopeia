#ifndef KGNavigableMeshIntersectionFinder_H__
#define KGNavigableMeshIntersectionFinder_H__

#include "KGArrayMath.hh"
#include "KGBoundaryCalculator.hh"
#include "KGCube.hh"
#include "KGIdentitySet.hh"
#include "KGMeshNavigationNode.hh"
#include "KGNavigableMeshElement.hh"
#include "KGNavigableMeshElementContainer.hh"
#include "KGNavigableMeshTree.hh"
#include "KGPoint.hh"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <set>
#include <stack>
#include <utility>
#include <vector>


namespace KGeoBag
{

/**
*
*@file KGNavigableMeshIntersectionFinder.hh
*@class KGNavigableMeshIntersectionFinder
*@brief finds all the intersections of a line segment and a mesh
* does NOT necessarily find them in order of distance to line segment start point)
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Jul 17 11:29:45 EDT 2015 J. Barrett (barrettj@mit.edu) First Version
*
*/


class KGNavigableMeshIntersectionFinder : public KGNodeActor<KGMeshNavigationNode>
{
  public:
    KGNavigableMeshIntersectionFinder();
    ~KGNavigableMeshIntersectionFinder() override;

    void SetMeshElementContainer(KGNavigableMeshElementContainer* container)
    {
        fContainer = container;
    };

    void NearestPointOnLineSegment(const katrin::KThreeVector& aPoint, katrin::KThreeVector& aNearest,
                                   double& t) const;
    double LineSegmentDistanceToPoint(const katrin::KThreeVector& aPoint) const;
    bool LineSegmentIntersectsCube(const KGCube<KGMESH_DIM>* cube, double& distance) const;

    void SetLineSegment(const katrin::KThreeVector& start, const katrin::KThreeVector& end);
    bool HasIntersectionWithMesh() const;
    unsigned int GetNIntersections() const
    {
        return fIntersections.size();
    };
    void GetIntersections(std::vector<katrin::KThreeVector>* intersections) const;
    void GetIntersectedMeshElements(std::vector<const KGNavigableMeshElement*>* intersected_mesh_elements) const;

    void ApplyAction(KGMeshNavigationNode* node) override;

  private:
    struct ChildDistanceOrder
    {
        bool operator()(std::pair<KGMeshNavigationNode*, double> a, std::pair<KGMeshNavigationNode*, double> b)
        {
            return (a.second > b.second);  //this will sort them from farthest to nearest
        }
    };

    //mesh element container
    KGNavigableMeshElementContainer* fContainer;

    //parameters of the line segment
    katrin::KThreeVector fStartPoint;
    katrin::KThreeVector fEndPoint;
    katrin::KThreeVector fDirection;
    double fLength;

    //needed for recursive access to tree
    std::stack<KGMeshNavigationNode*> fNodeStack;
    KGMeshNavigationNode* fTempNode;

    ChildDistanceOrder fOrderingPredicate;
    std::vector<std::pair<KGMeshNavigationNode*, double>> fOrderedChildren;

    //intersection data
    bool fHaveIntersection;
    std::vector<katrin::KThreeVector> fIntersections;
    std::vector<const KGNavigableMeshElement*> fIntersectedElements;

    std::set<const KGNavigableMeshElement*> fCheckedElements;
};


}  // namespace KGeoBag

#endif /* end of include guard: KGNavigableMeshIntersectionFinder_H__ */
