#include "KGExtrudedArcSegmentSurfaceMesher.hh"

namespace KGeoBag
{

    KGExtrudedArcSegmentSurfaceMesher::KGExtrudedArcSegmentSurfaceMesher() :
            KGSimpleMesher()
    {
    }
    KGExtrudedArcSegmentSurfaceMesher::~KGExtrudedArcSegmentSurfaceMesher()
    {
    }

    void KGExtrudedArcSegmentSurfaceMesher::VisitExtrudedPathSurface( KGExtrudedArcSegmentSurface* aExtrudedArcSegmentSurface )
    {
        //create arc segment points
        OpenPoints tArcSegmentPoints;
        ArcSegmentToOpenPoints( aExtrudedArcSegmentSurface->Path().operator ->(), tArcSegmentPoints );

        //create extruded points
        FlatMesh tMeshPoints;
        OpenPointsExtrudedToFlatMesh( tArcSegmentPoints, aExtrudedArcSegmentSurface->ZMin(), aExtrudedArcSegmentSurface->ZMax(), aExtrudedArcSegmentSurface->ExtrudedMeshCount(), aExtrudedArcSegmentSurface->ExtrudedMeshPower(), tMeshPoints );

        //create mesh
        FlatMeshToTriangles( tMeshPoints );

        return;
    }

}
