#include "KGRotatedArcSegmentSurfaceMesher.hh"

namespace KGeoBag
{

KGRotatedArcSegmentSurfaceMesher::KGRotatedArcSegmentSurfaceMesher() : KGSimpleMesher() {}
KGRotatedArcSegmentSurfaceMesher::~KGRotatedArcSegmentSurfaceMesher() {}

void KGRotatedArcSegmentSurfaceMesher::VisitRotatedPathSurface(KGRotatedArcSegmentSurface* aRotatedArcSegmentSurface)
{
    //create arc segment points
    OpenPoints tArcSegmentPoints;
    ArcSegmentToOpenPoints(aRotatedArcSegmentSurface->Path().operator->(), tArcSegmentPoints);

    //create rotated points
    TubeMesh tMeshPoints;
    OpenPointsRotatedToTubeMesh(tArcSegmentPoints, aRotatedArcSegmentSurface->RotatedMeshCount(), tMeshPoints);

    //surgery
    bool tHasStart = false;
    KThreeVector tStartApex;
    if (aRotatedArcSegmentSurface->Path()->Start().Y() == 0.) {
        tHasStart = true;
        tStartApex.SetComponents(0., 0., aRotatedArcSegmentSurface->Path()->Start().X());
        tMeshPoints.fData.pop_front();
    }

    bool tHasEnd = false;
    KThreeVector tEndApex;
    if (aRotatedArcSegmentSurface->Path()->End().Y() == 0.) {
        tHasEnd = true;
        tEndApex.SetComponents(0., 0., aRotatedArcSegmentSurface->Path()->End().X());
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
