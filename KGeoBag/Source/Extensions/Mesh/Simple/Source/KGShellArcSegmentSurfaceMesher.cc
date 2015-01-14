#include "KGShellArcSegmentSurfaceMesher.hh"

namespace KGeoBag
{

    KGShellArcSegmentSurfaceMesher::KGShellArcSegmentSurfaceMesher() :
            KGSimpleMesher()
    {
    }
    KGShellArcSegmentSurfaceMesher::~KGShellArcSegmentSurfaceMesher()
    {
    }

    void KGShellArcSegmentSurfaceMesher::VisitShellPathSurface( KGShellArcSegmentSurface* aShellArcSegmentSurface )
    {std::cout << "visiting shell path" << std::endl;
        //create arc segment points
        OpenPoints tArcSegmentPoints;
        ArcSegmentToOpenPoints( aShellArcSegmentSurface->Path().operator ->(), tArcSegmentPoints );

        //create Shell points
        ShellMesh tMeshPoints;
        OpenPointsRotatedToShellMesh( tArcSegmentPoints, aShellArcSegmentSurface->ShellMeshCount(), aShellArcSegmentSurface->ShellMeshPower(), tMeshPoints, aShellArcSegmentSurface->AngleStart(), aShellArcSegmentSurface->AngleStop()  );

        //create mesh
        ShellMeshToTriangles( tMeshPoints );

        return;
    }

}
