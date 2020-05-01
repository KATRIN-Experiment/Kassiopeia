#include "KGRotatedPolyLoopSurfaceAxialMesher.hh"

namespace KGeoBag
{

KGRotatedPolyLoopSurfaceAxialMesher::KGRotatedPolyLoopSurfaceAxialMesher() : KGSimpleAxialMesher() {}
KGRotatedPolyLoopSurfaceAxialMesher::~KGRotatedPolyLoopSurfaceAxialMesher() {}

void KGRotatedPolyLoopSurfaceAxialMesher::VisitRotatedPathSurface(KGRotatedPolyLoopSurface* aRotatedPolyLoopSurface)
{
    //create poly loop points
    ClosedPoints tPolyLoopPoints;
    PolyLoopToClosedPoints(aRotatedPolyLoopSurface->Path().operator->(), tPolyLoopPoints);

    //create loops
    ClosedPointsToLoops(tPolyLoopPoints);

    return;
}

}  // namespace KGeoBag
