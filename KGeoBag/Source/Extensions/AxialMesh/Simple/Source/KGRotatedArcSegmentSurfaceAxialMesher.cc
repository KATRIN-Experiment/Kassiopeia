#include "KGRotatedArcSegmentSurfaceAxialMesher.hh"

namespace KGeoBag
{

KGRotatedArcSegmentSurfaceAxialMesher::KGRotatedArcSegmentSurfaceAxialMesher() : KGSimpleAxialMesher() {}
KGRotatedArcSegmentSurfaceAxialMesher::~KGRotatedArcSegmentSurfaceAxialMesher() {}

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
