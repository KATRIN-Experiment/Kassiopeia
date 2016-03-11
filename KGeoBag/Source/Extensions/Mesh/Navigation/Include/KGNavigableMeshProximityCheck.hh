#ifndef KGNavigableMeshProximityCheck_H__
#define KGNavigableMeshProximityCheck_H__

#include <vector>
#include <complex>
#include <stack>
#include <cmath>
#include <cstdlib>
#include <utility>

#include "KGArrayMath.hh"

#include "KGPoint.hh"
#include "KGBall.hh"
#include "KGCube.hh"
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
*@file KGNavigableMeshProximityCheck.hh
*@class KGNavigableMeshProximityCheck
*@brief finds the first intersection (closest to start point) of a line segment and a mesh
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Jul 17 11:29:45 EDT 2015 J. Barrett (barrettj@mit.edu) First Version
*
*/


class KGNavigableMeshProximityCheck: public KGNodeActor< KGMeshNavigationNode >
{
    public:
        KGNavigableMeshProximityCheck();
        virtual ~KGNavigableMeshProximityCheck();

        void SetMeshElementContainer(KGNavigableMeshElementContainer* container){fContainer = container;};

        void SetPointAndRadius(const KThreeVector& point, double radius);
        bool SphereIntersectsMesh() const {return fSphereIntersectsMesh;};

        bool BallIntersectsCube(const KGBall<KGMESH_DIM>& ball, const KGCube<KGMESH_DIM>& cube) const;
        bool CubeEnclosedByBall(const KGBall<KGMESH_DIM>& ball, const KGCube<KGMESH_DIM>& cube) const;

        virtual void ApplyAction(KGMeshNavigationNode* node);

    private:

        //mesh element container
        KGNavigableMeshElementContainer* fContainer;

        //stack space and functions for tree traversal
        unsigned int fDefaultStackSize;
        unsigned int fStackReallocateLimit;
        typedef KGMeshNavigationNode* KGMeshNavigationNodePtr;
        KGMeshNavigationNodePtr* fPreallocatedStackTopPtr;
        std::vector< KGMeshNavigationNode* > fPreallocatedStack;
        unsigned int fStackSize;

        inline void CheckStackSize()
        {
            if(fStackSize >= fStackReallocateLimit)
            {
                fPreallocatedStack.resize(3*fStackSize);
                fStackReallocateLimit = 2*fStackSize;
            }
        };

        static const double fCubeLengthToRadius;

        //parameters of the line segment
        KThreeVector fPoint;
        double fRadius;
        KGBall<KGMESH_DIM> fBall;

        //list of leaf nodes that contain elements which might lie within bounding ball
        std::vector< KGMeshNavigationNode* > fLeafNodes;

        //mesh enters bounding ball or not
        bool fSphereIntersectsMesh;

};


}

#endif /* end of include guard: KGNavigableMeshProximityCheck_H__ */
