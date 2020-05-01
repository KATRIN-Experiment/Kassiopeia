#include "KGRotatedArcSegmentSpaceMesher.hh"

namespace KGeoBag
{

KGRotatedArcSegmentSpaceMesher::KGRotatedArcSegmentSpaceMesher() : KGSimpleMesher() {}
KGRotatedArcSegmentSpaceMesher::~KGRotatedArcSegmentSpaceMesher() {}

void KGRotatedArcSegmentSpaceMesher::VisitRotatedOpenPathSpace(KGRotatedArcSegmentSpace* aRotatedArcSegmentSpace)
{
    //create line segment points
    OpenPoints tArcSegmentPoints;
    ArcSegmentToOpenPoints(aRotatedArcSegmentSpace->Path().operator->(), tArcSegmentPoints);

    //create rotated points
    TubeMesh tMeshPoints;
    OpenPointsRotatedToTubeMesh(tArcSegmentPoints, aRotatedArcSegmentSpace->RotatedMeshCount(), tMeshPoints);

    //make room for ends
    tMeshPoints.fData.pop_front();
    tMeshPoints.fData.pop_back();

    //surgery
    KThreeVector tStartApex;
    if (aRotatedArcSegmentSpace->StartPath()) {
        //create start circle points
        ClosedPoints tStartCirclePoints;
        CircleToClosedPoints(aRotatedArcSegmentSpace->StartPath().operator->(), tStartCirclePoints);

        //create start flattened mesh points
        TubeMesh tStartMeshPoints;
        ClosedPointsFlattenedToTubeMeshAndApex(tStartCirclePoints,
                                               aRotatedArcSegmentSpace->StartPath()->Centroid(),
                                               aRotatedArcSegmentSpace->Path()->Start().X(),
                                               aRotatedArcSegmentSpace->FlattenedMeshCount(),
                                               aRotatedArcSegmentSpace->FlattenedMeshPower(),
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
        tStartApex.SetComponents(0., 0., aRotatedArcSegmentSpace->Path()->Start().X());
    }

    KThreeVector tEndApex;
    if (aRotatedArcSegmentSpace->EndPath()) {
        //create end circle points
        ClosedPoints tEndCirclePoints;
        CircleToClosedPoints(aRotatedArcSegmentSpace->EndPath().operator->(), tEndCirclePoints);

        //create end flattened mesh points
        TubeMesh tEndMeshPoints;
        ClosedPointsFlattenedToTubeMeshAndApex(tEndCirclePoints,
                                               aRotatedArcSegmentSpace->EndPath()->Centroid(),
                                               aRotatedArcSegmentSpace->Path()->End().X(),
                                               aRotatedArcSegmentSpace->FlattenedMeshCount(),
                                               aRotatedArcSegmentSpace->FlattenedMeshPower(),
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
        tEndApex.SetComponents(0., 0., aRotatedArcSegmentSpace->Path()->End().X());
    }

    //lay triangles on the mesh
    TubeMeshToTriangles(tStartApex, tMeshPoints, tEndApex);

    return;
}

}  // namespace KGeoBag
