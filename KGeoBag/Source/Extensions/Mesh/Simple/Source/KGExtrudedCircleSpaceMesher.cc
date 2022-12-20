#include "KGExtrudedCircleSpaceMesher.hh"

namespace KGeoBag
{

KGExtrudedCircleSpaceMesher::KGExtrudedCircleSpaceMesher() = default;
KGExtrudedCircleSpaceMesher::~KGExtrudedCircleSpaceMesher() = default;

void KGExtrudedCircleSpaceMesher::VisitExtrudedClosedPathSpace(KGExtrudedCircleSpace* aExtrudedCircleSpace)
{
    //create circle points
    ClosedPoints tCirclePoints;
    CircleToClosedPoints(aExtrudedCircleSpace->Path().operator->(), tCirclePoints);

    //create extruded points
    TubeMesh tMeshPoints;
    ClosedPointsExtrudedToTubeMesh(tCirclePoints,
                                   aExtrudedCircleSpace->ZMin(),
                                   aExtrudedCircleSpace->ZMax(),
                                   aExtrudedCircleSpace->ExtrudedMeshCount(),
                                   aExtrudedCircleSpace->ExtrudedMeshPower(),
                                   tMeshPoints);

    //create start flattened mesh points
    TubeMesh tStartMeshPoints;
    katrin::KThreeVector tStartApex;
    ClosedPointsFlattenedToTubeMeshAndApex(tCirclePoints,
                                           aExtrudedCircleSpace->Path()->Centroid(),
                                           aExtrudedCircleSpace->ZMin(),
                                           aExtrudedCircleSpace->FlattenedMeshCount(),
                                           aExtrudedCircleSpace->FlattenedMeshPower(),
                                           tStartMeshPoints,
                                           tStartApex);

    //create end flattened mesh points
    TubeMesh tEndMeshPoints;
    katrin::KThreeVector tEndApex;
    ClosedPointsFlattenedToTubeMeshAndApex(tCirclePoints,
                                           aExtrudedCircleSpace->Path()->Centroid(),
                                           aExtrudedCircleSpace->ZMax(),
                                           aExtrudedCircleSpace->FlattenedMeshCount(),
                                           aExtrudedCircleSpace->FlattenedMeshPower(),
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
