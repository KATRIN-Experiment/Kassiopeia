#include "KGExtrudedLineSegmentSurfaceMesher.hh"

namespace KGeoBag
{

KGExtrudedLineSegmentSurfaceMesher::KGExtrudedLineSegmentSurfaceMesher() : KGSimpleMesher() {}
KGExtrudedLineSegmentSurfaceMesher::~KGExtrudedLineSegmentSurfaceMesher() {}

void KGExtrudedLineSegmentSurfaceMesher::VisitExtrudedPathSurface(
    KGExtrudedLineSegmentSurface* aExtrudedLineSegmentSurface)
{
    //create line segment points
    OpenPoints tLineSegmentPoints;
    LineSegmentToOpenPoints(aExtrudedLineSegmentSurface->Path().operator->(), tLineSegmentPoints);

    //create extruded points
    FlatMesh tMeshPoints;
    OpenPointsExtrudedToFlatMesh(tLineSegmentPoints,
                                 aExtrudedLineSegmentSurface->ZMin(),
                                 aExtrudedLineSegmentSurface->ZMax(),
                                 aExtrudedLineSegmentSurface->ExtrudedMeshCount(),
                                 aExtrudedLineSegmentSurface->ExtrudedMeshPower(),
                                 tMeshPoints);

    //create mesh
    FlatMeshToTriangles(tMeshPoints);

    return;
}

}  // namespace KGeoBag
