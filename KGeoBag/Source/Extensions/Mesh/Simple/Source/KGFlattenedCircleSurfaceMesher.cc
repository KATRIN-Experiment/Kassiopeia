#include "KGFlattenedCircleSurfaceMesher.hh"

namespace KGeoBag
{

    KGFlattenedCircleSurfaceMesher::KGFlattenedCircleSurfaceMesher() :
            KGSimpleMesher()
    {
    }
    KGFlattenedCircleSurfaceMesher::~KGFlattenedCircleSurfaceMesher()
    {
    }

    void KGFlattenedCircleSurfaceMesher::VisitFlattenedClosedPathSurface( KGFlattenedCircleSurface* aFlattenedCircleSurface )
    {
        //create circle points
        ClosedPoints tCirclePoints;
        CircleToClosedPoints( aFlattenedCircleSurface->Path().operator ->(), tCirclePoints );

        //create flattened points
        KThreeVector tApexPoint;
        TubeMesh tMeshPoints;
        ClosedPointsFlattenedToTubeMeshAndApex( tCirclePoints, aFlattenedCircleSurface->Path()->Centroid(), aFlattenedCircleSurface->Z(), aFlattenedCircleSurface->FlattenedMeshCount(), aFlattenedCircleSurface->FlattenedMeshPower(), tMeshPoints, tApexPoint );

        //create mesh
        TubeMeshToTriangles( tMeshPoints, tApexPoint );

        return;
    }

}
