#ifndef KGeoBag_KGAxialMesher_hh_
#define KGeoBag_KGAxialMesher_hh_

#include "KGCore.hh"

#include "KGRotatedLineSegmentSurfaceAxialMesher.hh"
#include "KGRotatedArcSegmentSurfaceAxialMesher.hh"
#include "KGRotatedPolyLineSurfaceAxialMesher.hh"
#include "KGRotatedCircleSurfaceAxialMesher.hh"
#include "KGRotatedPolyLoopSurfaceAxialMesher.hh"
#include "KGRotatedLineSegmentSpaceAxialMesher.hh"
#include "KGRotatedArcSegmentSpaceAxialMesher.hh"
#include "KGRotatedPolyLineSpaceAxialMesher.hh"
#include "KGRotatedCircleSpaceAxialMesher.hh"
#include "KGRotatedPolyLoopSpaceAxialMesher.hh"

namespace KGeoBag
{
    class KGAxialMesher :
        virtual public KGRotatedLineSegmentSurfaceAxialMesher,
        virtual public KGRotatedArcSegmentSurfaceAxialMesher,
        virtual public KGRotatedPolyLineSurfaceAxialMesher,
        virtual public KGRotatedCircleSurfaceAxialMesher,
        virtual public KGRotatedPolyLoopSurfaceAxialMesher,
        virtual public KGRotatedLineSegmentSpaceAxialMesher,
        virtual public KGRotatedArcSegmentSpaceAxialMesher,
        virtual public KGRotatedPolyLineSpaceAxialMesher,
        virtual public KGRotatedCircleSpaceAxialMesher,
        virtual public KGRotatedPolyLoopSpaceAxialMesher
    {
        public:
            using KGAxialMesherBase::VisitExtendedSurface;
            using KGAxialMesherBase::VisitExtendedSpace;

            using KGRotatedLineSegmentSurfaceAxialMesher::VisitRotatedPathSurface;
            using KGRotatedArcSegmentSurfaceAxialMesher::VisitRotatedPathSurface;
            using KGRotatedPolyLineSurfaceAxialMesher::VisitRotatedPathSurface;
            using KGRotatedCircleSurfaceAxialMesher::VisitRotatedPathSurface;
            using KGRotatedPolyLoopSurfaceAxialMesher::VisitRotatedPathSurface;
            using KGRotatedLineSegmentSpaceAxialMesher::VisitRotatedOpenPathSpace;
            using KGRotatedArcSegmentSpaceAxialMesher::VisitRotatedOpenPathSpace;
            using KGRotatedPolyLineSpaceAxialMesher::VisitRotatedOpenPathSpace;
            using KGRotatedCircleSpaceAxialMesher::VisitRotatedClosedPathSpace;
            using KGRotatedPolyLoopSpaceAxialMesher::VisitRotatedClosedPathSpace;

        public:
            KGAxialMesher();
            virtual ~KGAxialMesher();

    };
}

#endif
