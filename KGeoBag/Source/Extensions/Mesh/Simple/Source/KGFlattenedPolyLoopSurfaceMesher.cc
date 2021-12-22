#include "KGFlattenedPolyLoopSurfaceMesher.hh"

namespace KGeoBag
{

KGFlattenedPolyLoopSurfaceMesher::KGFlattenedPolyLoopSurfaceMesher() = default;
KGFlattenedPolyLoopSurfaceMesher::~KGFlattenedPolyLoopSurfaceMesher() = default;

void KGFlattenedPolyLoopSurfaceMesher::VisitFlattenedClosedPathSurface(
    KGFlattenedPolyLoopSurface* aFlattenedPolyLoopSurface)
{
    //create circle points
    ClosedPoints tPolyLoopPoints;
    PolyLoopToClosedPoints(aFlattenedPolyLoopSurface->Path().operator->(), tPolyLoopPoints);

    //create flattened points
    katrin::KThreeVector tApexPoint;
    TubeMesh tMeshPoints;
    ClosedPointsFlattenedToTubeMeshAndApex(tPolyLoopPoints,
                                           aFlattenedPolyLoopSurface->Path()->Centroid(),
                                           aFlattenedPolyLoopSurface->Z(),
                                           aFlattenedPolyLoopSurface->FlattenedMeshCount(),
                                           aFlattenedPolyLoopSurface->FlattenedMeshPower(),
                                           tMeshPoints,
                                           tApexPoint);

    //create mesh
    TubeMeshToTriangles(tMeshPoints, tApexPoint);

    return;
}

}  // namespace KGeoBag
