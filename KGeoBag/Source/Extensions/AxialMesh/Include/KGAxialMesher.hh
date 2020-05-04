#ifndef KGeoBag_KGAxialMesher_hh_
#define KGeoBag_KGAxialMesher_hh_

#include "KGCore.hh"
#include "KGRotatedArcSegmentSpaceAxialMesher.hh"
#include "KGRotatedArcSegmentSurfaceAxialMesher.hh"
#include "KGRotatedCircleSpaceAxialMesher.hh"
#include "KGRotatedCircleSurfaceAxialMesher.hh"
#include "KGRotatedLineSegmentSpaceAxialMesher.hh"
#include "KGRotatedLineSegmentSurfaceAxialMesher.hh"
#include "KGRotatedPolyLineSpaceAxialMesher.hh"
#include "KGRotatedPolyLineSurfaceAxialMesher.hh"
#include "KGRotatedPolyLoopSpaceAxialMesher.hh"
#include "KGRotatedPolyLoopSurfaceAxialMesher.hh"

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
    using KGAxialMesherBase::VisitExtendedSpace;
    using KGAxialMesherBase::VisitExtendedSurface;

    using KGRotatedArcSegmentSpaceAxialMesher::VisitRotatedOpenPathSpace;
    using KGRotatedArcSegmentSurfaceAxialMesher::VisitRotatedPathSurface;
    using KGRotatedCircleSpaceAxialMesher::VisitRotatedClosedPathSpace;
    using KGRotatedCircleSurfaceAxialMesher::VisitRotatedPathSurface;
    using KGRotatedLineSegmentSpaceAxialMesher::VisitRotatedOpenPathSpace;
    using KGRotatedLineSegmentSurfaceAxialMesher::VisitRotatedPathSurface;
    using KGRotatedPolyLineSpaceAxialMesher::VisitRotatedOpenPathSpace;
    using KGRotatedPolyLineSurfaceAxialMesher::VisitRotatedPathSurface;
    using KGRotatedPolyLoopSpaceAxialMesher::VisitRotatedClosedPathSpace;
    using KGRotatedPolyLoopSurfaceAxialMesher::VisitRotatedPathSurface;

  public:
    KGAxialMesher();
    ~KGAxialMesher() override;
};
}  // namespace KGeoBag

#endif
