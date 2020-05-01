#include "KGRotatedLineSegmentSurfaceAxialMesher.hh"

namespace KGeoBag
{

KGRotatedLineSegmentSurfaceAxialMesher::KGRotatedLineSegmentSurfaceAxialMesher() : KGSimpleAxialMesher() {}
KGRotatedLineSegmentSurfaceAxialMesher::~KGRotatedLineSegmentSurfaceAxialMesher() {}

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
