#include "KGShellPolyLoopSurfaceMesher.hh"

namespace KGeoBag
{

KGShellPolyLoopSurfaceMesher::KGShellPolyLoopSurfaceMesher() : KGSimpleMesher() {}
KGShellPolyLoopSurfaceMesher::~KGShellPolyLoopSurfaceMesher() {}

void KGShellPolyLoopSurfaceMesher::VisitShellPathSurface(KGShellPolyLoopSurface* aShellPolyLoopSurface)
{
    //create poly loop points
    ClosedPoints tPolyLoopPoints;
    PolyLoopToClosedPoints(aShellPolyLoopSurface->Path().operator->(), tPolyLoopPoints);

    //create shell points
    ShellMesh tMeshPoints;
    ClosedPointsRotatedToShellMesh(tPolyLoopPoints,
                                   aShellPolyLoopSurface->ShellMeshCount(),
                                   aShellPolyLoopSurface->ShellMeshPower(),
                                   tMeshPoints,
                                   aShellPolyLoopSurface->AngleStart(),
                                   aShellPolyLoopSurface->AngleStop());

    //create mesh
    ClosedShellMeshToTriangles(tMeshPoints);

    return;
}

}  // namespace KGeoBag
