#include "KGRotatedCircleSurfaceAxialMesher.hh"

namespace KGeoBag
{

KGRotatedCircleSurfaceAxialMesher::KGRotatedCircleSurfaceAxialMesher() = default;
KGRotatedCircleSurfaceAxialMesher::~KGRotatedCircleSurfaceAxialMesher() = default;

void KGRotatedCircleSurfaceAxialMesher::VisitRotatedPathSurface(KGRotatedCircleSurface* aRotatedCircleSurface)
{
    //create poly line points
    ClosedPoints tCirclePoints;
    CircleToClosedPoints(aRotatedCircleSurface->Path().operator->(), tCirclePoints);

    //create loops
    ClosedPointsToLoops(tCirclePoints);

    return;
}

}  // namespace KGeoBag
