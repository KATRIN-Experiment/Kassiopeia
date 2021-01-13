#include "KGRotatedLineSegmentSurfaceAxialMesher.hh"

namespace KGeoBag
{

KGRotatedLineSegmentSurfaceAxialMesher::KGRotatedLineSegmentSurfaceAxialMesher() = default;
KGRotatedLineSegmentSurfaceAxialMesher::~KGRotatedLineSegmentSurfaceAxialMesher() = default;

void KGRotatedLineSegmentSurfaceAxialMesher::VisitRotatedPathSurface(
    KGRotatedLineSegmentSurface* aRotatedLineSegmentSurface)
{
    //create line segment points
    OpenPoints tLineSegmentPoints;
    LineSegmentToOpenPoints(aRotatedLineSegmentSurface->Path().operator->(), tLineSegmentPoints);

    //create loops
    OpenPointsToLoops(tLineSegmentPoints);

    return;
}

}  // namespace KGeoBag
