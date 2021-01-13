#include "KGExtrudedCircleSurfaceMesher.hh"

namespace KGeoBag
{

KGExtrudedCircleSurfaceMesher::KGExtrudedCircleSurfaceMesher() = default;
KGExtrudedCircleSurfaceMesher::~KGExtrudedCircleSurfaceMesher() = default;

void KGExtrudedCircleSurfaceMesher::VisitExtrudedPathSurface(KGExtrudedCircleSurface* aExtrudedCircleSurface)
{
    //create poly line points
    ClosedPoints tCirclePoints;
    CircleToClosedPoints(aExtrudedCircleSurface->Path().operator->(), tCirclePoints);

    //create rotated points
    TubeMesh tMeshPoints;
    ClosedPointsExtrudedToTubeMesh(tCirclePoints,
                                   aExtrudedCircleSurface->ZMin(),
                                   aExtrudedCircleSurface->ZMax(),
                                   aExtrudedCircleSurface->ExtrudedMeshCount(),
                                   aExtrudedCircleSurface->ExtrudedMeshPower(),
                                   tMeshPoints);

    //create mesh
    TubeMeshToTriangles(tMeshPoints);

    return;
}

}  // namespace KGeoBag
