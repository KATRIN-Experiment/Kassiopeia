#ifndef KGNavigableMeshIntersectionFinder_H__
#define KGNavigableMeshIntersectionFinder_H__

#include <vector>
#include <complex>
#include <stack>
#include <cmath>
#include <cstdlib>
#include <utility>
#include <algorithm>
#include <set>

#include "KGArrayMath.hh"

#include "KGCube.hh"
#include "KGPoint.hh"
#include "KGIdentitySet.hh"

#include "KGNavigableMeshElement.hh"
#include "KGMeshNavigationNode.hh"
#include "KGNavigableMeshTree.hh"
#include "KGNavigableMeshElementContainer.hh"

#include "KGBoundaryCalculator.hh"


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


class KGNavigableMeshIntersectionFinder: public KGNodeActor< KGMeshNavigationNode >
{
    public:
        KGNavigableMeshIntersectionFinder();
        virtual ~KGNavigableMeshIntersectionFinder();

        void SetMeshElementContainer(KGNavigableMeshElementContainer* container){fContainer = container;};

        void NearestPointOnLineSegment(const KThreeVector& aPoint, KThreeVector& aNearest, double& t) const;
        double LineSegmentDistanceToPoint(const KThreeVector& aPoint) const;
        bool LineSegmentIntersectsCube(const KGCube<KGMESH_DIM>* cube, double& distance) const;

        void SetLineSegment(const KThreeVector& start, const KThreeVector& end);
        bool HasIntersectionWithMesh() const;
        unsigned int GetNIntersections() const {return fIntersections.size();};
        void GetIntersections( std::vector< KThreeVector >* intersections) const;
        void GetIntersectedMeshElements( std::vector< const KGNavigableMeshElement* >* intersected_mesh_elements) const;

        virtual void ApplyAction(KGMeshNavigationNode* node);

    private:

        struct ChildDistanceOrder
        {
            bool operator() (std::pair< KGMeshNavigationNode*, double > a, std::pair< KGMeshNavigationNode*, double > b)
            {
                return (a.second > b.second ); //this will sort them from farthest to nearest

            }
        };

        //mesh element container
        KGNavigableMeshElementContainer* fContainer;

        //parameters of the line segment
        KThreeVector fStartPoint;
        KThreeVector fEndPoint;
        KThreeVector fDirection;
        double fLength;

        //needed for recursive access to tree
        std::stack< KGMeshNavigationNode* > fNodeStack;
        KGMeshNavigationNode* fTempNode;

        ChildDistanceOrder fOrderingPredicate;
        std::vector< std::pair< KGMeshNavigationNode*, double > > fOrderedChildren;

        //intersection data
        bool fHaveIntersection;
        std::vector< KThreeVector > fIntersections;
        std::vector< const KGNavigableMeshElement* > fIntersectedElements;

        std::set< const KGNavigableMeshElement* > fCheckedElements;

};


}

#endif /* end of include guard: KGNavigableMeshIntersectionFinder_H__ */
