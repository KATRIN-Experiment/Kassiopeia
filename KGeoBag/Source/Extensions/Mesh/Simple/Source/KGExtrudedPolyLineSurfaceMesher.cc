#include "KGExtrudedPolyLineSurfaceMesher.hh"

namespace KGeoBag
{

KGExtrudedPolyLineSurfaceMesher::KGExtrudedPolyLineSurfaceMesher() = default;
KGExtrudedPolyLineSurfaceMesher::~KGExtrudedPolyLineSurfaceMesher() = default;

void KGExtrudedPolyLineSurfaceMesher::VisitExtrudedPathSurface(KGExtrudedPolyLineSurface* aExtrudedPolyLineSurface)
{
    //create poly line points
    OpenPoints tPolyLinePoints;
    PolyLineToOpenPoints(aExtrudedPolyLineSurface->Path().operator->(), tPolyLinePoints);

    //create extruded points
    FlatMesh tMeshPoints;
    OpenPointsExtrudedToFlatMesh(tPolyLinePoints,
                                 aExtrudedPolyLineSurface->ZMin(),
                                 aExtrudedPolyLineSurface->ZMax(),
                                 aExtrudedPolyLineSurface->ExtrudedMeshCount(),
                                 aExtrudedPolyLineSurface->ExtrudedMeshPower(),
                                 tMeshPoints);

    //create mesh
    FlatMeshToTriangles(tMeshPoints);

    return;
}

}  // namespace KGeoBag
