#ifndef KGNavigableMeshFirstIntersectionFinder_H__
#define KGNavigableMeshFirstIntersectionFinder_H__

#include <vector>
#include <complex>
#include <stack>
#include <cmath>
#include <cstdlib>
#include <utility>

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


class KGNavigableMeshFirstIntersectionFinder: public KGNodeActor< KGMeshNavigationNode >
{
    public:
        KGNavigableMeshFirstIntersectionFinder();
        virtual ~KGNavigableMeshFirstIntersectionFinder();

        void SetMeshElementContainer(KGNavigableMeshElementContainer* container){fContainer = container;};
        void SetSilent(){fVerbose = false;};
        void SetVerbose(){fVerbose = true;};

        void NearestPointOnLineSegment(const KThreeVector& aPoint, KThreeVector& aNearest, double& t) const;
        double LineSegmentDistanceToPoint(const KThreeVector& aPoint) const;
        bool LineSegmentIntersectsCube(const KGCube<KGMESH_DIM>& cube, double& distance) const;


        void SetLineSegment(const KThreeVector& start, const KThreeVector& end);
        bool HasIntersectionWithMesh() const;
        KThreeVector GetIntersection() const;
        const KGNavigableMeshElement* GetIntersectedMeshElement() const {return fIntersectedElement;};

        virtual void ApplyAction(KGMeshNavigationNode* node);

    private:

        bool fVerbose;

        //mesh element container
        KGNavigableMeshElementContainer* fContainer;

        struct ChildDistanceOrder
        {
            bool operator() (std::pair< KGMeshNavigationNode*, double > a, std::pair< KGMeshNavigationNode*, double > b)
            {
                return (a.second > b.second ); //this will sort them from farthest to nearest

            }
        };


        //stack space and functions for tree traversal
        unsigned int fDefaultStackSize;
        unsigned int fStackReallocateLimit;
        typedef KGMeshNavigationNode* KGMeshNavigationNodePtr;
        KGMeshNavigationNodePtr* fPreallocatedStackTopPtr;
        std::vector< KGMeshNavigationNode* > fPreallocatedStack;
        unsigned int fStackSize;

        //sort the intersected child node by distance from line segment start
        static void SortOctreeNodes(unsigned int n_nodes, std::pair< KGMeshNavigationNode*, double >* nodes);

        inline void CheckStackSize()
        {
            if(fStackSize >= fStackReallocateLimit)
            {
                fPreallocatedStack.resize(3*fStackSize);
                fStackReallocateLimit = 2*fStackSize;
            }
        };

        //parameters of the line segment
        KThreeVector fStartPoint;
        KThreeVector fEndPoint;
        KThreeVector fDirection;
        double fLength;

        KGMeshNavigationNode* fTempNode;
        std::pair< KGMeshNavigationNode*, double > fOrderedChildren[8];

        //intersection data
        bool fHaveIntersection;
        KThreeVector fFirstIntersection;
        KGNavigableMeshElement* fIntersectedElement;

};


}

#endif /* end of include guard: KGNavigableMeshFirstIntersectionFinder_H__ */
