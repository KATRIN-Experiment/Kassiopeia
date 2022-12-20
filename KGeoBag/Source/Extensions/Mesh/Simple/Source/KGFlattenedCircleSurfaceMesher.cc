#include "KGFlattenedCircleSurfaceMesher.hh"

namespace KGeoBag
{

KGFlattenedCircleSurfaceMesher::KGFlattenedCircleSurfaceMesher() = default;
KGFlattenedCircleSurfaceMesher::~KGFlattenedCircleSurfaceMesher() = default;

void KGFlattenedCircleSurfaceMesher::VisitFlattenedClosedPathSurface(KGFlattenedCircleSurface* aFlattenedCircleSurface)
{
    //create circle points
    ClosedPoints tCirclePoints;
    CircleToClosedPoints(aFlattenedCircleSurface->Path().operator->(), tCirclePoints);

    //create flattened points
    katrin::KThreeVector tApexPoint;
    TubeMesh tMeshPoints;
    ClosedPointsFlattenedToTubeMeshAndApex(tCirclePoints,
                                           aFlattenedCircleSurface->Path()->Centroid(),
                                           aFlattenedCircleSurface->Z(),
                                           aFlattenedCircleSurface->FlattenedMeshCount(),
                                           aFlattenedCircleSurface->FlattenedMeshPower(),
                                           tMeshPoints,
                                           tApexPoint);

    //create mesh
    TubeMeshToTriangles(tMeshPoints, tApexPoint);

    return;
}

}  // namespace KGeoBag
