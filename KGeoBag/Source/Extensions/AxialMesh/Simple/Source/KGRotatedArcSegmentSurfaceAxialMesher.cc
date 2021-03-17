#include "KGRotatedArcSegmentSurfaceAxialMesher.hh"

namespace KGeoBag
{

KGRotatedArcSegmentSurfaceAxialMesher::KGRotatedArcSegmentSurfaceAxialMesher() = default;
KGRotatedArcSegmentSurfaceAxialMesher::~KGRotatedArcSegmentSurfaceAxialMesher() = default;

void KGRotatedArcSegmentSurfaceAxialMesher::VisitRotatedPathSurface(
    KGRotatedArcSegmentSurface* aRotatedArcSegmentSurface)
{
    //create arc segment points
    OpenPoints tArcSegmentPoints;
    ArcSegmentToOpenPoints(aRotatedArcSegmentSurface->Path().operator->(), tArcSegmentPoints);

    //create loops
    OpenPointsToLoops(tArcSegmentPoints);

    return;
}

}  // namespace KGeoBag
