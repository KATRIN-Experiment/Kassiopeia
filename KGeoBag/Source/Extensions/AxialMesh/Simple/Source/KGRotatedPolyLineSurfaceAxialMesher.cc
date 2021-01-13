#include "KGRotatedPolyLineSurfaceAxialMesher.hh"

namespace KGeoBag
{

KGRotatedPolyLineSurfaceAxialMesher::KGRotatedPolyLineSurfaceAxialMesher() = default;
KGRotatedPolyLineSurfaceAxialMesher::~KGRotatedPolyLineSurfaceAxialMesher() = default;

void KGRotatedPolyLineSurfaceAxialMesher::VisitRotatedPathSurface(KGRotatedPolyLineSurface* aRotatedPolyLineSurface)
{
    //create poly line points
    OpenPoints tPolyLinePoints;
    PolyLineToOpenPoints(aRotatedPolyLineSurface->Path().operator->(), tPolyLinePoints);

    //create loops
    OpenPointsToLoops(tPolyLinePoints);

    return;
}

}  // namespace KGeoBag
