#ifndef KGeoBag_KGSimpleMesher_hh_
#define KGeoBag_KGSimpleMesher_hh_

#include "KGMesherBase.hh"

#include "KGPlanarLineSegment.hh"
#include "KGPlanarArcSegment.hh"
#include "KGPlanarPolyLine.hh"
#include "KGPlanarCircle.hh"
#include "KGPlanarPolyLoop.hh"

#include <deque>
using std::deque;

namespace KGeoBag
{

    class KGSimpleMesher :
        virtual public KGMesherBase
    {
        public:
            using KGMesherBase::VisitExtendedSurface;
            using KGMesherBase::VisitExtendedSpace;

        public:
            KGSimpleMesher();
            virtual ~KGSimpleMesher();

            //**********
            //data types
            //**********

        protected:
            class Partition
            {
                public:
                    typedef double Value;
                    typedef deque< Value > Set;
                    typedef Set::iterator It;
                    typedef Set::const_iterator CIt;

                public:
                    Set fData;
            };

            class Points
            {
                public:
                    typedef KTwoVector Element;
                    typedef deque< Element > Set;
                    typedef Set::iterator It;
                    typedef Set::const_iterator CIt;

                public:
                    Set fData;
            };

            class OpenPoints :
                public Points
            {
            };

            class ClosedPoints :
                public Points
            {
            };

            class Mesh
            {
                public:
                    typedef KThreeVector Element;
                    typedef deque< KThreeVector > Group;
                    typedef Group::iterator GroupIt;
                    typedef Group::const_iterator GroupCIt;
                    typedef deque< Group > Set;
                    typedef Set::iterator SetIt;
                    typedef Set::const_iterator SetCIt;

                public:
                    Set fData;
            };

            class FlatMesh :
                public Mesh
            {
            };

            class TubeMesh :
                public Mesh
            {
            };

            class ShellMesh :
                public Mesh
            {
            };

            class TorusMesh :
                public Mesh
            {
            };

            //*******************
            //partition functions
            //*******************

        protected:
            void SymmetricPartition( const double& aStart, const double& aStop, const unsigned int& aCount, const double& aPower, Partition& aPartition );
            void ForwardPartition( const double& aStart, const double& aStop, const unsigned int& aCount, const double& aPower, Partition& aPartition );
            void BackwardPartition( const double& aStart, const double& aStop, const unsigned int& aCount, const double& aPower, Partition& aPartition );

            //****************
            //points functions
            //****************

        protected:
            void LineSegmentToOpenPoints( const KGPlanarLineSegment* aLineSegment, OpenPoints& aPoints );
            void ArcSegmentToOpenPoints( const KGPlanarArcSegment* anArcSegment, OpenPoints& aPoints );
            void PolyLineToOpenPoints( const KGPlanarPolyLine* aPolyLine, OpenPoints& aPoints );
            void CircleToClosedPoints( const KGPlanarCircle* aCircle, ClosedPoints& aPoints );
            void PolyLoopToClosedPoints( const KGPlanarPolyLoop* aPolyLoop, ClosedPoints& aPoints );

            //**************
            //mesh functions
            //**************

        protected:
            void ClosedPointsFlattenedToTubeMeshAndApex( const ClosedPoints& aPoints, const KTwoVector& aCentroid, const double& aZ, const unsigned int& aCount, const double& aPower, TubeMesh& aMesh, KThreeVector& anApex );
            void OpenPointsRotatedToTubeMesh( const OpenPoints& aPoints, const unsigned int& aCount, TubeMesh& aMesh );
            void OpenPointsRotatedToShellMesh( const OpenPoints& aPoints, const unsigned int& aCount, const double& aPower, ShellMesh& aMesh, const double& aAngleStart, const double& aAngleStop );
            void ClosedPointsRotatedToShellMesh( const ClosedPoints& aPoints, const unsigned int& aCount, const double& aPower, ShellMesh& aMesh, const double& aAngleStart, const double& aAngleStop );
            void ClosedPointsRotatedToTorusMesh( const ClosedPoints& aPoints, const unsigned int& aCount, TorusMesh& aMesh );
            void OpenPointsExtrudedToFlatMesh( const OpenPoints& aPoints, const double& aZMin, const double& aZMax, const unsigned int& aCount, const double& aPower, FlatMesh& aMesh );
            void ClosedPointsExtrudedToTubeMesh( const ClosedPoints& aPoints, const double& aZMin, const double& aZMax, const unsigned int& aCount, const double& aPower, TubeMesh& aMesh );

            //*********************
            //tesselation functions
            //*********************

        protected:
            void FlatMeshToTriangles( const FlatMesh& aMesh );
            void TubeMeshToTriangles( const TubeMesh& aMesh );
            void TubeMeshToTriangles( const TubeMesh& aMesh, const KThreeVector& anApexEnd );
            void TubeMeshToTriangles( const KThreeVector& anApexStart, const TubeMesh& aMesh );
            void TubeMeshToTriangles( const KThreeVector& anApexStart, const TubeMesh& aMesh, const KThreeVector& anApexEnd );
            void ShellMeshToTriangles( const ShellMesh& aMesh );
            void ClosedShellMeshToTriangles( const ShellMesh& aMesh );
            void TorusMeshToTriangles( const TorusMesh& aMesh );

            //*****************
            //triangle function
            //*****************

        protected:
            void Triangle( const KThreeVector& aFirst, const KThreeVector& aSecond, const KThreeVector& aThird );
    };

}

#endif
