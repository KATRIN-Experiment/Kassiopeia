#include "KGExtrudedPolyLineSurfaceMesher.hh"

namespace KGeoBag
{

    KGExtrudedPolyLineSurfaceMesher::KGExtrudedPolyLineSurfaceMesher() :
            KGSimpleMesher()
    {
    }
    KGExtrudedPolyLineSurfaceMesher::~KGExtrudedPolyLineSurfaceMesher()
    {
    }

    void KGExtrudedPolyLineSurfaceMesher::VisitExtrudedPathSurface( KGExtrudedPolyLineSurface* aExtrudedPolyLineSurface )
    {
        //create poly line points
        OpenPoints tPolyLinePoints;
        PolyLineToOpenPoints( aExtrudedPolyLineSurface->Path().operator ->(), tPolyLinePoints );

        //create extruded points
        FlatMesh tMeshPoints;
        OpenPointsExtrudedToFlatMesh( tPolyLinePoints, aExtrudedPolyLineSurface->ZMin(), aExtrudedPolyLineSurface->ZMax(), aExtrudedPolyLineSurface->ExtrudedMeshCount(), aExtrudedPolyLineSurface->ExtrudedMeshPower(), tMeshPoints );

        //create mesh
        FlatMeshToTriangles( tMeshPoints );

        return;
    }

}
