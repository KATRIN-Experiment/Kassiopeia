#include "KGExtrudedPolyLoopSurfaceMesher.hh"

namespace KGeoBag
{

    KGExtrudedPolyLoopSurfaceMesher::KGExtrudedPolyLoopSurfaceMesher() :
            KGSimpleMesher()
    {
    }
    KGExtrudedPolyLoopSurfaceMesher::~KGExtrudedPolyLoopSurfaceMesher()
    {
    }

    void KGExtrudedPolyLoopSurfaceMesher::VisitExtrudedPathSurface( KGExtrudedPolyLoopSurface* aExtrudedPolyLoopSurface )
    {
        //create poly loop points
        ClosedPoints tPolyLoopPoints;
        PolyLoopToClosedPoints( aExtrudedPolyLoopSurface->Path().operator ->(), tPolyLoopPoints );

        //create rotated points
        TubeMesh tMeshPoints;
        ClosedPointsExtrudedToTubeMesh( tPolyLoopPoints, aExtrudedPolyLoopSurface->ZMin(), aExtrudedPolyLoopSurface->ZMax(), aExtrudedPolyLoopSurface->ExtrudedMeshCount(), aExtrudedPolyLoopSurface->ExtrudedMeshPower(), tMeshPoints );

        //create mesh
        TubeMeshToTriangles( tMeshPoints );

        return;
    }

}
