#include "KGRotatedPolyLineSpaceMesher.hh"

namespace KGeoBag
{

KGRotatedPolyLineSpaceMesher::KGRotatedPolyLineSpaceMesher() : KGSimpleMesher() {}
KGRotatedPolyLineSpaceMesher::~KGRotatedPolyLineSpaceMesher() {}

void KGRotatedPolyLineSpaceMesher::VisitRotatedOpenPathSpace(KGRotatedPolyLineSpace* aRotatedPolyLineSpace)
{
    //create line segment points
    OpenPoints tPolyLinePoints;
    PolyLineToOpenPoints(aRotatedPolyLineSpace->Path().operator->(), tPolyLinePoints);

    //create rotated points
    TubeMesh tMeshPoints;
    OpenPointsRotatedToTubeMesh(tPolyLinePoints, aRotatedPolyLineSpace->RotatedMeshCount(), tMeshPoints);

    //make room for ends
    tMeshPoints.fData.pop_front();
    tMeshPoints.fData.pop_back();

    //surgery
    KThreeVector tStartApex;
    if (aRotatedPolyLineSpace->StartPath()) {
        //create start circle points
        ClosedPoints tStartCirclePoints;
        CircleToClosedPoints(aRotatedPolyLineSpace->StartPath().operator->(), tStartCirclePoints);

        //create start flattened mesh points
        TubeMesh tStartMeshPoints;
        ClosedPointsFlattenedToTubeMeshAndApex(tStartCirclePoints,
                                               aRotatedPolyLineSpace->StartPath()->Centroid(),
                                               aRotatedPolyLineSpace->Path()->Start().X(),
                                               aRotatedPolyLineSpace->FlattenedMeshCount(),
                                               aRotatedPolyLineSpace->FlattenedMeshPower(),
                                               tStartMeshPoints,
                                               tStartApex);

        //stitch circle mesh onto main mesh
        auto tCircleIt = tStartMeshPoints.fData.begin();
        while (tCircleIt != tStartMeshPoints.fData.end()) {
            tMeshPoints.fData.push_front(*tCircleIt);
            ++tCircleIt;
        }
    }
    else {
        //otherwise make the apex by hand
        tStartApex.SetComponents(0., 0., aRotatedPolyLineSpace->Path()->Start().X());
    }

    KThreeVector tEndApex;
    if (aRotatedPolyLineSpace->EndPath()) {
        //create end circle points
        ClosedPoints tEndCirclePoints;
        CircleToClosedPoints(aRotatedPolyLineSpace->EndPath().operator->(), tEndCirclePoints);

        //create end flattened mesh points
        TubeMesh tEndMeshPoints;
        ClosedPointsFlattenedToTubeMeshAndApex(tEndCirclePoints,
                                               aRotatedPolyLineSpace->EndPath()->Centroid(),
                                               aRotatedPolyLineSpace->Path()->End().X(),
                                               aRotatedPolyLineSpace->FlattenedMeshCount(),
                                               aRotatedPolyLineSpace->FlattenedMeshPower(),
                                               tEndMeshPoints,
                                               tEndApex);

        auto tCircleIt = tEndMeshPoints.fData.begin();
        while (tCircleIt != tEndMeshPoints.fData.end()) {
            tMeshPoints.fData.push_back(*tCircleIt);
            ++tCircleIt;
        }
    }
    else {
        //otherwise make the apex by hand
        tEndApex.SetComponents(0., 0., aRotatedPolyLineSpace->Path()->End().X());
    }

    //lay triangles on the mesh
    TubeMeshToTriangles(tStartApex, tMeshPoints, tEndApex);

    return;
}

}  // namespace KGeoBag
