#include "KGRotatedPolyLineSurfaceMesher.hh"

namespace KGeoBag
{

KGRotatedPolyLineSurfaceMesher::KGRotatedPolyLineSurfaceMesher() : KGSimpleMesher() {}
KGRotatedPolyLineSurfaceMesher::~KGRotatedPolyLineSurfaceMesher() {}

void KGRotatedPolyLineSurfaceMesher::VisitRotatedPathSurface(KGRotatedPolyLineSurface* aRotatedPolyLineSurface)
{
    //create poly line points
    OpenPoints tPolyLinePoints;
    PolyLineToOpenPoints(aRotatedPolyLineSurface->Path().operator->(), tPolyLinePoints);

    //create rotated points
    TubeMesh tMeshPoints;
    OpenPointsRotatedToTubeMesh(tPolyLinePoints, aRotatedPolyLineSurface->RotatedMeshCount(), tMeshPoints);

    //surgery
    bool tHasStart = false;
    KThreeVector tStartApex;
    if (aRotatedPolyLineSurface->Path()->Start().Y() == 0.) {
        tHasStart = true;
        tStartApex.SetComponents(0., 0., aRotatedPolyLineSurface->Path()->Start().X());
        tMeshPoints.fData.pop_front();
    }

    bool tHasEnd = false;
    KThreeVector tEndApex;
    if (aRotatedPolyLineSurface->Path()->End().Y() == 0.) {
        tHasEnd = true;
        tEndApex.SetComponents(0., 0., aRotatedPolyLineSurface->Path()->End().X());
        tMeshPoints.fData.pop_back();
    }

    //create mesh
    if (tHasStart == true) {
        if (tHasEnd == true) {
            TubeMeshToTriangles(tStartApex, tMeshPoints, tEndApex);
        }
        else {
            TubeMeshToTriangles(tStartApex, tMeshPoints);
        }
    }
    else {
        if (tHasEnd == true) {
            TubeMeshToTriangles(tMeshPoints, tEndApex);
        }
        else {
            TubeMeshToTriangles(tMeshPoints);
        }
    }

    return;
}

}  // namespace KGeoBag
