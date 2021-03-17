#include "KGRotatedPolyLoopSurfaceMesher.hh"

namespace KGeoBag
{

KGRotatedPolyLoopSurfaceMesher::KGRotatedPolyLoopSurfaceMesher() = default;
KGRotatedPolyLoopSurfaceMesher::~KGRotatedPolyLoopSurfaceMesher() = default;

void KGRotatedPolyLoopSurfaceMesher::VisitRotatedPathSurface(KGRotatedPolyLoopSurface* aRotatedPolyLoopSurface)
{
    //create poly loop points
    ClosedPoints tPolyLoopPoints;
    PolyLoopToClosedPoints(aRotatedPolyLoopSurface->Path().operator->(), tPolyLoopPoints);

    //create rotated points
    TorusMesh tMeshPoints;
    ClosedPointsRotatedToTorusMesh(tPolyLoopPoints, aRotatedPolyLoopSurface->RotatedMeshCount(), tMeshPoints);

    //create mesh
    TorusMeshToTriangles(tMeshPoints);

    return;
}

}  // namespace KGeoBag
