#include "KGShellLineSegmentSurfaceMesher.hh"

namespace KGeoBag
{

KGShellLineSegmentSurfaceMesher::KGShellLineSegmentSurfaceMesher() = default;
KGShellLineSegmentSurfaceMesher::~KGShellLineSegmentSurfaceMesher() = default;

void KGShellLineSegmentSurfaceMesher::VisitShellPathSurface(KGShellLineSegmentSurface* aShellLineSegmentSurface)
{
    //create line segment points
    OpenPoints tLineSegmentPoints;
    LineSegmentToOpenPoints(aShellLineSegmentSurface->Path().operator->(), tLineSegmentPoints);

    //create Shell points
    ShellMesh tMeshPoints;
    OpenPointsRotatedToShellMesh(tLineSegmentPoints,
                                 aShellLineSegmentSurface->ShellMeshCount(),
                                 aShellLineSegmentSurface->ShellMeshPower(),
                                 tMeshPoints,
                                 aShellLineSegmentSurface->AngleStart(),
                                 aShellLineSegmentSurface->AngleStop());

    //create mesh
    ShellMeshToTriangles(tMeshPoints);

    return;
}

}  // namespace KGeoBag
