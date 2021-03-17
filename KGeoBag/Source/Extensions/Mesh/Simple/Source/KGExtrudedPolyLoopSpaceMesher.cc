#include "KGExtrudedPolyLoopSpaceMesher.hh"

namespace KGeoBag
{

KGExtrudedPolyLoopSpaceMesher::KGExtrudedPolyLoopSpaceMesher() = default;
KGExtrudedPolyLoopSpaceMesher::~KGExtrudedPolyLoopSpaceMesher() = default;

void KGExtrudedPolyLoopSpaceMesher::VisitExtrudedClosedPathSpace(KGExtrudedPolyLoopSpace* aExtrudedPolyLoopSpace)
{
    //create circle points
    ClosedPoints tPolyLoopPoints;
    PolyLoopToClosedPoints(aExtrudedPolyLoopSpace->Path().operator->(), tPolyLoopPoints);

    //create extruded points
    TubeMesh tMeshPoints;
    ClosedPointsExtrudedToTubeMesh(tPolyLoopPoints,
                                   aExtrudedPolyLoopSpace->ZMin(),
                                   aExtrudedPolyLoopSpace->ZMax(),
                                   aExtrudedPolyLoopSpace->ExtrudedMeshCount(),
                                   aExtrudedPolyLoopSpace->ExtrudedMeshPower(),
                                   tMeshPoints);

    //create start flattened mesh points
    TubeMesh tStartMeshPoints;
    KThreeVector tStartApex;
    ClosedPointsFlattenedToTubeMeshAndApex(tPolyLoopPoints,
                                           aExtrudedPolyLoopSpace->Path()->Centroid(),
                                           aExtrudedPolyLoopSpace->ZMin(),
                                           aExtrudedPolyLoopSpace->FlattenedMeshCount(),
                                           aExtrudedPolyLoopSpace->FlattenedMeshPower(),
                                           tStartMeshPoints,
                                           tStartApex);

    //create end flattened mesh points
    TubeMesh tEndMeshPoints;
    KThreeVector tEndApex;
    ClosedPointsFlattenedToTubeMeshAndApex(tPolyLoopPoints,
                                           aExtrudedPolyLoopSpace->Path()->Centroid(),
                                           aExtrudedPolyLoopSpace->ZMax(),
                                           aExtrudedPolyLoopSpace->FlattenedMeshCount(),
                                           aExtrudedPolyLoopSpace->FlattenedMeshPower(),
                                           tEndMeshPoints,
                                           tEndApex);

    //surgery
    tMeshPoints.fData.pop_front();
    for (auto& tStartIt : tStartMeshPoints.fData) {
        tMeshPoints.fData.push_front(tStartIt);
    }

    tMeshPoints.fData.pop_back();
    for (auto& tEndIt : tEndMeshPoints.fData) {
        tMeshPoints.fData.push_back(tEndIt);
    }

    //create mesh
    TubeMeshToTriangles(tStartApex, tMeshPoints, tEndApex);

    return;
}

}  // namespace KGeoBag
