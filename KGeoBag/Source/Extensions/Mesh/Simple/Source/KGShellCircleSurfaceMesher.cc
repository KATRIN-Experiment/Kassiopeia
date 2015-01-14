#include "KGShellCircleSurfaceMesher.hh"

namespace KGeoBag
{

    KGShellCircleSurfaceMesher::KGShellCircleSurfaceMesher() :
            KGSimpleMesher()
    {
    }
    KGShellCircleSurfaceMesher::~KGShellCircleSurfaceMesher()
    {
    }

    void KGShellCircleSurfaceMesher::VisitShellPathSurface( KGShellCircleSurface* aShellCircleSurface )
    {
        //create poly line points
        ClosedPoints tCirclePoints;
        CircleToClosedPoints( aShellCircleSurface->Path().operator ->(), tCirclePoints );

        //create shell points
        ShellMesh tMeshPoints;
        ClosedPointsRotatedToShellMesh( tCirclePoints, aShellCircleSurface->ShellMeshCount(),aShellCircleSurface->ShellMeshPower(), tMeshPoints , aShellCircleSurface->AngleStart(), aShellCircleSurface->AngleStop()  );

        //create mesh
        ClosedShellMeshToTriangles( tMeshPoints );

        return;
    }

}
