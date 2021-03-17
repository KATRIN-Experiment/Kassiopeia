#include "KGRotatedCircleSurfaceMesher.hh"

namespace KGeoBag
{

KGRotatedCircleSurfaceMesher::KGRotatedCircleSurfaceMesher() = default;
KGRotatedCircleSurfaceMesher::~KGRotatedCircleSurfaceMesher() = default;

void KGRotatedCircleSurfaceMesher::VisitRotatedPathSurface(KGRotatedCircleSurface* aRotatedCircleSurface)
{
    //create poly line points
    ClosedPoints tCirclePoints;
    CircleToClosedPoints(aRotatedCircleSurface->Path().operator->(), tCirclePoints);

    //create rotated points
    TorusMesh tMeshPoints;
    ClosedPointsRotatedToTorusMesh(tCirclePoints, aRotatedCircleSurface->RotatedMeshCount(), tMeshPoints);

    //create mesh
    TorusMeshToTriangles(tMeshPoints);

    return;
}

}  // namespace KGeoBag
