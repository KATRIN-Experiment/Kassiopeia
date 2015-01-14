#include "KGFlattenedPolyLoopSurfaceMesher.hh"

namespace KGeoBag
{

    KGFlattenedPolyLoopSurfaceMesher::KGFlattenedPolyLoopSurfaceMesher() :
            KGSimpleMesher()
    {
    }
    KGFlattenedPolyLoopSurfaceMesher::~KGFlattenedPolyLoopSurfaceMesher()
    {
    }

    void KGFlattenedPolyLoopSurfaceMesher::VisitFlattenedClosedPathSurface( KGFlattenedPolyLoopSurface* aFlattenedPolyLoopSurface )
    {
        //create circle points
        ClosedPoints tPolyLoopPoints;
        PolyLoopToClosedPoints( aFlattenedPolyLoopSurface->Path().operator ->(), tPolyLoopPoints );

        //create flattened points
        KThreeVector tApexPoint;
        TubeMesh tMeshPoints;
        ClosedPointsFlattenedToTubeMeshAndApex( tPolyLoopPoints, aFlattenedPolyLoopSurface->Path()->Centroid(), aFlattenedPolyLoopSurface->Z(), aFlattenedPolyLoopSurface->FlattenedMeshCount(), aFlattenedPolyLoopSurface->FlattenedMeshPower(), tMeshPoints, tApexPoint );

        //create mesh
        TubeMeshToTriangles( tMeshPoints, tApexPoint );

        return;
    }

}
