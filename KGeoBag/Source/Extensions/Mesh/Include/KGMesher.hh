#ifndef KGeoBag_KGMesher_hh_
#define KGeoBag_KGMesher_hh_

#include "KGCore.hh"

#include "KGBeamSurfaceMesher.hh"
#include "KGBoxMesher.hh"
#include "KGConicSectPortHousingSurfaceMesher.hh"
#include "KGCylinderMesher.hh"
#include "KGExtrudedSurfaceMesher.hh"
#include "KGConicalWireArrayMesher.hh"
#include "KGPortHousingSurfaceMesher.hh"
#include "KGRodSurfaceMesher.hh"
#include "KGRotatedSurfaceMesher.hh"

#include "KGFlattenedCircleSurfaceMesher.hh"
#include "KGFlattenedPolyLoopSurfaceMesher.hh"
#include "KGRotatedLineSegmentSurfaceMesher.hh"
#include "KGRotatedArcSegmentSurfaceMesher.hh"
#include "KGRotatedPolyLineSurfaceMesher.hh"
#include "KGRotatedCircleSurfaceMesher.hh"
#include "KGRotatedPolyLoopSurfaceMesher.hh"
#include "KGShellLineSegmentSurfaceMesher.hh"
#include "KGShellArcSegmentSurfaceMesher.hh"
#include "KGShellPolyLineSurfaceMesher.hh"
#include "KGShellPolyLoopSurfaceMesher.hh"
#include "KGShellCircleSurfaceMesher.hh"
#include "KGExtrudedLineSegmentSurfaceMesher.hh"
#include "KGExtrudedArcSegmentSurfaceMesher.hh"
#include "KGExtrudedPolyLineSurfaceMesher.hh"
#include "KGExtrudedCircleSurfaceMesher.hh"
#include "KGExtrudedPolyLoopSurfaceMesher.hh"
#include "KGRotatedLineSegmentSpaceMesher.hh"
#include "KGRotatedArcSegmentSpaceMesher.hh"
#include "KGRotatedPolyLineSpaceMesher.hh"
#include "KGRotatedCircleSpaceMesher.hh"
#include "KGRotatedPolyLoopSpaceMesher.hh"
#include "KGExtrudedCircleSpaceMesher.hh"
#include "KGExtrudedPolyLoopSpaceMesher.hh"

namespace KGeoBag
{
    class KGMesher :
        virtual public KGBoxMesher,
        virtual public KGCylinderMesher,
        virtual public KGRotatedSurfaceMesher,
        virtual public KGExtrudedSurfaceMesher,
        virtual public KGRodSurfaceMesher,
        virtual public KGBeamSurfaceMesher,
        virtual public KGConicalWireArrayMesher,
        virtual public KGPortHousingSurfaceMesher,
        virtual public KGConicSectPortHousingSurfaceMesher,
        virtual public KGFlattenedCircleSurfaceMesher,
        virtual public KGFlattenedPolyLoopSurfaceMesher,
        virtual public KGRotatedLineSegmentSurfaceMesher,
        virtual public KGRotatedArcSegmentSurfaceMesher,
        virtual public KGRotatedPolyLineSurfaceMesher,
        virtual public KGRotatedCircleSurfaceMesher,
        virtual public KGRotatedPolyLoopSurfaceMesher,
        virtual public KGShellLineSegmentSurfaceMesher,
        virtual public KGShellArcSegmentSurfaceMesher,
        virtual public KGShellPolyLineSurfaceMesher,
        virtual public KGShellPolyLoopSurfaceMesher,
        virtual public KGShellCircleSurfaceMesher,
        virtual public KGExtrudedLineSegmentSurfaceMesher,
        virtual public KGExtrudedArcSegmentSurfaceMesher,
        virtual public KGExtrudedPolyLineSurfaceMesher,
        virtual public KGExtrudedCircleSurfaceMesher,
        virtual public KGExtrudedPolyLoopSurfaceMesher,
        virtual public KGRotatedLineSegmentSpaceMesher,
        virtual public KGRotatedArcSegmentSpaceMesher,
        virtual public KGRotatedPolyLineSpaceMesher,
        virtual public KGRotatedCircleSpaceMesher,
        virtual public KGRotatedPolyLoopSpaceMesher,
        virtual public KGExtrudedCircleSpaceMesher,
        virtual public KGExtrudedPolyLoopSpaceMesher
    {
        public:
            using KGMesherBase::VisitExtendedSurface;
            using KGMesherBase::VisitExtendedSpace;

            using KGBoxMesher::VisitBox;
            using KGCylinderMesher::VisitCylinder;
            using KGRotatedSurfaceMesher::VisitWrappedSurface;
            using KGExtrudedSurfaceMesher::VisitWrappedSurface;
            using KGRodSurfaceMesher::VisitWrappedSurface;
            using KGBeamSurfaceMesher::VisitWrappedSurface;
            using KGConicalWireArrayMesher::VisitWrappedSurface;
            using KGPortHousingSurfaceMesher::VisitWrappedSurface;
            using KGConicSectPortHousingSurfaceMesher::VisitWrappedSurface;

            using KGFlattenedCircleSurfaceMesher::VisitFlattenedClosedPathSurface;
            using KGFlattenedPolyLoopSurfaceMesher::VisitFlattenedClosedPathSurface;
            using KGRotatedLineSegmentSurfaceMesher::VisitRotatedPathSurface;
            using KGRotatedArcSegmentSurfaceMesher::VisitRotatedPathSurface;
            using KGRotatedPolyLineSurfaceMesher::VisitRotatedPathSurface;
            using KGRotatedCircleSurfaceMesher::VisitRotatedPathSurface;
            using KGRotatedPolyLoopSurfaceMesher::VisitRotatedPathSurface;
            using KGShellLineSegmentSurfaceMesher::VisitShellPathSurface;
            using KGShellArcSegmentSurfaceMesher::VisitShellPathSurface;
            using KGShellPolyLineSurfaceMesher::VisitShellPathSurface;
            using KGShellPolyLoopSurfaceMesher::VisitShellPathSurface;
            using KGShellCircleSurfaceMesher::VisitShellPathSurface;
            using KGExtrudedLineSegmentSurfaceMesher::VisitExtrudedPathSurface;
            using KGExtrudedArcSegmentSurfaceMesher::VisitExtrudedPathSurface;
            using KGExtrudedPolyLineSurfaceMesher::VisitExtrudedPathSurface;
            using KGExtrudedCircleSurfaceMesher::VisitExtrudedPathSurface;
            using KGExtrudedPolyLoopSurfaceMesher::VisitExtrudedPathSurface;
            using KGRotatedLineSegmentSpaceMesher::VisitRotatedOpenPathSpace;
            using KGRotatedArcSegmentSpaceMesher::VisitRotatedOpenPathSpace;
            using KGRotatedPolyLineSpaceMesher::VisitRotatedOpenPathSpace;
            using KGRotatedCircleSpaceMesher::VisitRotatedClosedPathSpace;
            using KGRotatedPolyLoopSpaceMesher::VisitRotatedClosedPathSpace;
            using KGExtrudedCircleSpaceMesher::VisitExtrudedClosedPathSpace;
            using KGExtrudedPolyLoopSpaceMesher::VisitExtrudedClosedPathSpace;

        public:
            KGMesher();
            virtual ~KGMesher();

    };
}

#endif /* KGDETERMINISTICMESHER_DEF */
