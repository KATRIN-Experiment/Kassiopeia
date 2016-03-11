#include "KGShellPolyLineSurfaceMesher.hh"

namespace KGeoBag
{

    KGShellPolyLineSurfaceMesher::KGShellPolyLineSurfaceMesher() :
    KGSimpleMesher()
    {
    }
    KGShellPolyLineSurfaceMesher::~KGShellPolyLineSurfaceMesher()
    {
    }

    void KGShellPolyLineSurfaceMesher::VisitShellPathSurface( KGShellPolyLineSurface* aShellPolyLineSurface )
    {
        //create poly line points
        OpenPoints tPolyLinePoints;
        PolyLineToOpenPoints( aShellPolyLineSurface->Path().operator ->(), tPolyLinePoints );

        //create shell points
        ShellMesh tMeshPoints;
        OpenPointsRotatedToShellMesh( tPolyLinePoints, aShellPolyLineSurface->ShellMeshCount(), aShellPolyLineSurface->ShellMeshPower(),  tMeshPoints, aShellPolyLineSurface->AngleStart(), aShellPolyLineSurface->AngleStop()   );

        //create mesh
        ShellMeshToTriangles( tMeshPoints );
        

        return;
    }

}
