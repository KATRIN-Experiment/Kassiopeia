#ifndef KGeoBag_KGMesher_hh_
#define KGeoBag_KGMesher_hh_

#include "KGBeamSurfaceMesher.hh"
#include "KGBoxMesher.hh"
#include "KGCircleWireMesher.hh"
#include "KGCircularWirePinsMesher.hh"
#include "KGConicSectPortHousingSurfaceMesher.hh"
#include "KGConicalWireArrayMesher.hh"
#include "KGCore.hh"
#include "KGCylinderMesher.hh"
#include "KGExtrudedArcSegmentSurfaceMesher.hh"
#include "KGExtrudedCircleSpaceMesher.hh"
#include "KGExtrudedCircleSurfaceMesher.hh"
#include "KGExtrudedLineSegmentSurfaceMesher.hh"
#include "KGExtrudedPolyLineSurfaceMesher.hh"
#include "KGExtrudedPolyLoopSpaceMesher.hh"
#include "KGExtrudedPolyLoopSurfaceMesher.hh"
#include "KGExtrudedSurfaceMesher.hh"
#include "KGFlattenedCircleSurfaceMesher.hh"
#include "KGFlattenedPolyLoopSurfaceMesher.hh"
#include "KGLinearWireGridMesher.hh"
#include "KGPortHousingSurfaceMesher.hh"
#include "KGQuadraticWireGridMesher.hh"
#include "KGRodSurfaceMesher.hh"
#include "KGRotatedArcSegmentSpaceMesher.hh"
#include "KGRotatedArcSegmentSurfaceMesher.hh"
#include "KGRotatedCircleSpaceMesher.hh"
#include "KGRotatedCircleSurfaceMesher.hh"
#include "KGRotatedLineSegmentSpaceMesher.hh"
#include "KGRotatedLineSegmentSurfaceMesher.hh"
#include "KGRotatedPolyLineSpaceMesher.hh"
#include "KGRotatedPolyLineSurfaceMesher.hh"
#include "KGRotatedPolyLoopSpaceMesher.hh"
#include "KGRotatedPolyLoopSurfaceMesher.hh"
#include "KGRotatedSurfaceMesher.hh"
#include "KGShellArcSegmentSurfaceMesher.hh"
#include "KGShellCircleSurfaceMesher.hh"
#include "KGShellLineSegmentSurfaceMesher.hh"
#include "KGShellPolyLineSurfaceMesher.hh"
#include "KGShellPolyLoopSurfaceMesher.hh"
#include "KGStlFileSurfaceMesher.hh"
#include "KGStlFileSpaceMesher.hh"

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
    virtual public KGStlFileSurfaceMesher,
    virtual public KGLinearWireGridMesher,
    virtual public KGQuadraticWireGridMesher,
    virtual public KGCircleWireMesher,
    virtual public KGCircularWirePinsMesher,
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
    virtual public KGExtrudedPolyLoopSpaceMesher,
    virtual public KGStlFileSpaceMesher
{
  public:
    using KGMesherBase::VisitExtendedSpace;
    using KGMesherBase::VisitExtendedSurface;

    using KGBeamSurfaceMesher::VisitWrappedSurface;
    using KGBoxMesher::VisitBox;
    using KGCircleWireMesher::VisitWrappedSurface;
    using KGCircularWirePinsMesher::VisitWrappedSurface;
    using KGConicalWireArrayMesher::VisitWrappedSurface;
    using KGConicSectPortHousingSurfaceMesher::VisitWrappedSurface;
    using KGCylinderMesher::VisitCylinder;
    using KGExtrudedSurfaceMesher::VisitWrappedSurface;
    using KGLinearWireGridMesher::VisitWrappedSurface;
    using KGPortHousingSurfaceMesher::VisitWrappedSurface;
    using KGQuadraticWireGridMesher::VisitWrappedSurface;
    using KGRodSurfaceMesher::VisitWrappedSurface;
    using KGRotatedSurfaceMesher::VisitWrappedSurface;
    using KGStlFileSurfaceMesher::VisitWrappedSurface;

    using KGExtrudedArcSegmentSurfaceMesher::VisitExtrudedPathSurface;
    using KGExtrudedCircleSpaceMesher::VisitExtrudedClosedPathSpace;
    using KGExtrudedCircleSurfaceMesher::VisitExtrudedPathSurface;
    using KGExtrudedLineSegmentSurfaceMesher::VisitExtrudedPathSurface;
    using KGExtrudedPolyLineSurfaceMesher::VisitExtrudedPathSurface;
    using KGExtrudedPolyLoopSpaceMesher::VisitExtrudedClosedPathSpace;
    using KGExtrudedPolyLoopSurfaceMesher::VisitExtrudedPathSurface;
    using KGFlattenedCircleSurfaceMesher::VisitFlattenedClosedPathSurface;
    using KGFlattenedPolyLoopSurfaceMesher::VisitFlattenedClosedPathSurface;
    using KGRotatedArcSegmentSpaceMesher::VisitRotatedOpenPathSpace;
    using KGRotatedArcSegmentSurfaceMesher::VisitRotatedPathSurface;
    using KGRotatedCircleSpaceMesher::VisitRotatedClosedPathSpace;
    using KGRotatedCircleSurfaceMesher::VisitRotatedPathSurface;
    using KGRotatedLineSegmentSpaceMesher::VisitRotatedOpenPathSpace;
    using KGRotatedLineSegmentSurfaceMesher::VisitRotatedPathSurface;
    using KGRotatedPolyLineSpaceMesher::VisitRotatedOpenPathSpace;
    using KGRotatedPolyLineSurfaceMesher::VisitRotatedPathSurface;
    using KGRotatedPolyLoopSpaceMesher::VisitRotatedClosedPathSpace;
    using KGRotatedPolyLoopSurfaceMesher::VisitRotatedPathSurface;
    using KGShellArcSegmentSurfaceMesher::VisitShellPathSurface;
    using KGShellCircleSurfaceMesher::VisitShellPathSurface;
    using KGShellLineSegmentSurfaceMesher::VisitShellPathSurface;
    using KGShellPolyLineSurfaceMesher::VisitShellPathSurface;
    using KGShellPolyLoopSurfaceMesher::VisitShellPathSurface;
    using KGStlFileSpaceMesher::VisitWrappedSpace;

  public:
    KGMesher();
    ~KGMesher() override;
};
}  // namespace KGeoBag

#endif /* KGDETERMINISTICMESHER_DEF */
