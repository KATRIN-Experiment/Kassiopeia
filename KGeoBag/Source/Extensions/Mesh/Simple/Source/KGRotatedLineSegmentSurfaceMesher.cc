#include "KGRotatedLineSegmentSurfaceMesher.hh"

namespace KGeoBag
{

KGRotatedLineSegmentSurfaceMesher::KGRotatedLineSegmentSurfaceMesher() : KGSimpleMesher() {}
KGRotatedLineSegmentSurfaceMesher::~KGRotatedLineSegmentSurfaceMesher() {}

void KGRotatedLineSegmentSurfaceMesher::VisitRotatedPathSurface(KGRotatedLineSegmentSurface* aRotatedLineSegmentSurface)
{
    //create line segment points
    OpenPoints tLineSegmentPoints;
    LineSegmentToOpenPoints(aRotatedLineSegmentSurface->Path().operator->(), tLineSegmentPoints);

    //create rotated points
    TubeMesh tMeshPoints;
    OpenPointsRotatedToTubeMesh(tLineSegmentPoints, aRotatedLineSegmentSurface->RotatedMeshCount(), tMeshPoints);

    //surgery
    bool tHasStart = false;
    KThreeVector tStartApex;
    if (aRotatedLineSegmentSurface->Path()->Start().Y() == 0.) {
        tHasStart = true;
        tStartApex.SetComponents(0., 0., aRotatedLineSegmentSurface->Path()->Start().X());
        tMeshPoints.fData.pop_front();
    }

    bool tHasEnd = false;
    KThreeVector tEndApex;
    if (aRotatedLineSegmentSurface->Path()->End().Y() == 0.) {
        tHasEnd = true;
        tEndApex.SetComponents(0., 0., aRotatedLineSegmentSurface->Path()->End().X());
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
