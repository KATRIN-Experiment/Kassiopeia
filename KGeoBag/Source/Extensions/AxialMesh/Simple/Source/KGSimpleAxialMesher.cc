#include "KGSimpleAxialMesher.hh"

#include "KConst.h"
#include "KGAxialMeshLoop.hh"
#include "KGAxialMeshMessage.hh"

#include <cmath>

namespace KGeoBag
{

KGSimpleAxialMesher::KGSimpleAxialMesher() {}
KGSimpleAxialMesher::~KGSimpleAxialMesher() {}

//*******************
//partition functions
//*******************

void KGSimpleAxialMesher::SymmetricPartition(const double& aStart, const double& aStop, const unsigned int& aCount,
                                             const double& aPower, Partition& aPartition)
{
    double tPower = aPower;
    double tStart = aStart;
    double tStop = aStop;
    double tMid = .5 * (tStop - tStart);
    double tY;
    double tX;

    aPartition.fData.clear();
    for (unsigned int tIndex = 0; tIndex <= aCount; tIndex++) {
        tY = (double) (tIndex) / (double) (aCount);
        if (tY < 0.5) {
            tX = tStart + tMid * pow(2. * tY, tPower);
        }
        else {
            tX = tStop - tMid * pow(2. - 2. * tY, tPower);
        }
        aPartition.fData.push_back(tX);
    }

    return;
}
void KGSimpleAxialMesher::ForwardPartition(const double& aStart, const double& aStop, const unsigned int& aCount,
                                           const double& aPower, Partition& aPartition)
{
    double tPower = aPower;
    double tStart = aStart;
    double tStop = aStop;
    double tLength = tStop - tStart;
    double tY;
    double tX;

    aPartition.fData.clear();
    for (unsigned int tIndex = 0; tIndex <= aCount; tIndex++) {
        tY = (double) (tIndex) / (double) (aCount);
        tX = tStart + tLength * pow(tY, tPower);
        aPartition.fData.push_back(tX);
    }

    return;
}
void KGSimpleAxialMesher::BackwardPartition(const double& aStart, const double& aStop, const unsigned int& aCount,
                                            const double& aPower, Partition& aPartition)
{
    double tPower = aPower;
    double tStart = aStart;
    double tStop = aStop;
    double tLength = tStop - tStart;
    double tY;
    double tX;

    aPartition.fData.clear();
    for (unsigned int tIndex = 0; tIndex <= aCount; tIndex++) {
        tY = (double) (tIndex) / (double) (aCount);
        tX = tStop - tLength * pow(1. - 1. * tY, tPower);
        aPartition.fData.push_back(tX);
    }

    return;
}

//****************
//points functions
//****************

void KGSimpleAxialMesher::LineSegmentToOpenPoints(const KGPlanarLineSegment* aLineSegment, OpenPoints& aPoints)
{
    Partition tPartition;
    Partition::It tPartitionIt;

    SymmetricPartition(0., aLineSegment->Length(), aLineSegment->MeshCount(), aLineSegment->MeshPower(), tPartition);

    aPoints.fData.clear();
    for (tPartitionIt = tPartition.fData.begin(); tPartitionIt != tPartition.fData.end(); tPartitionIt++) {
        aPoints.fData.push_back(aLineSegment->At(*tPartitionIt));
    }

    axialmeshmsg_debug("line segment partitioned into <" << aPoints.fData.size() << "> open points vertices" << eom);

    return;
}
void KGSimpleAxialMesher::ArcSegmentToOpenPoints(const KGPlanarArcSegment* anArcSegment, OpenPoints& aPoints)
{
    Partition tPartition;
    Partition::It tPartitionIt;

    SymmetricPartition(0., anArcSegment->Length(), anArcSegment->MeshCount(), 1., tPartition);

    aPoints.fData.clear();
    for (tPartitionIt = tPartition.fData.begin(); tPartitionIt != tPartition.fData.end(); tPartitionIt++) {
        aPoints.fData.push_back(anArcSegment->At(*tPartitionIt));
    }

    axialmeshmsg_debug("arc segment partitioned into <" << aPoints.fData.size() << "> open points vertices" << eom);

    return;
}
void KGSimpleAxialMesher::PolyLineToOpenPoints(const KGPlanarPolyLine* aPolyLine, OpenPoints& aPoints)
{
    const KGPlanarPolyLine::Set& tElements = aPolyLine->Elements();
    KGPlanarPolyLine::CIt tElementIt;
    const KGPlanarOpenPath* tElement;
    const KGPlanarLineSegment* tLineSegmentElement;
    const KGPlanarArcSegment* tArcSegmentElement;

    OpenPoints tSubPoints;
    for (tElementIt = tElements.begin(); tElementIt != tElements.end(); tElementIt++) {
        tElement = *tElementIt;

        tLineSegmentElement = dynamic_cast<const KGPlanarLineSegment*>(tElement);
        if (tLineSegmentElement != nullptr) {
            LineSegmentToOpenPoints(tLineSegmentElement, tSubPoints);
            aPoints.fData.insert(aPoints.fData.end(), tSubPoints.fData.begin(), --(tSubPoints.fData.end()));
            continue;
        }

        tArcSegmentElement = dynamic_cast<const KGPlanarArcSegment*>(tElement);
        if (tArcSegmentElement != nullptr) {
            ArcSegmentToOpenPoints(tArcSegmentElement, tSubPoints);
            aPoints.fData.insert(aPoints.fData.end(), tSubPoints.fData.begin(), --(tSubPoints.fData.end()));
            continue;
        }
    }

    aPoints.fData.push_back(aPolyLine->End());

    axialmeshmsg_debug("poly line partitioned into <" << aPoints.fData.size() << "> open points vertices" << eom);

    return;
}
void KGSimpleAxialMesher::CircleToClosedPoints(const KGPlanarCircle* aCircle, ClosedPoints& aPoints)
{
    Partition tPartition;
    Partition::It tPartitionIt;

    SymmetricPartition(0., aCircle->Length(), aCircle->MeshCount(), 1., tPartition);

    aPoints.fData.clear();
    for (tPartitionIt = tPartition.fData.begin(); tPartitionIt != --(tPartition.fData.end()); tPartitionIt++) {
        aPoints.fData.push_back(aCircle->At(*tPartitionIt));
    }

    axialmeshmsg_debug("circle partitioned into <" << aPoints.fData.size() << "> closed points vertices" << eom);

    return;
}
void KGSimpleAxialMesher::PolyLoopToClosedPoints(const KGPlanarPolyLoop* aPolyLoop, ClosedPoints& aPoints)
{
    const KGPlanarPolyLoop::Set& tElements = aPolyLoop->Elements();
    KGPlanarPolyLoop::CIt tElementIt;
    const KGPlanarOpenPath* tElement;
    const KGPlanarLineSegment* tLineSegmentElement;
    const KGPlanarArcSegment* tArcSegmentElement;

    OpenPoints tSubPoints;
    for (tElementIt = tElements.begin(); tElementIt != tElements.end(); tElementIt++) {
        tElement = *tElementIt;

        tLineSegmentElement = dynamic_cast<const KGPlanarLineSegment*>(tElement);
        if (tLineSegmentElement != nullptr) {
            LineSegmentToOpenPoints(tLineSegmentElement, tSubPoints);
            aPoints.fData.insert(aPoints.fData.end(), tSubPoints.fData.begin(), --(tSubPoints.fData.end()));
            continue;
        }

        tArcSegmentElement = dynamic_cast<const KGPlanarArcSegment*>(tElement);
        if (tArcSegmentElement != nullptr) {
            ArcSegmentToOpenPoints(tArcSegmentElement, tSubPoints);
            aPoints.fData.insert(aPoints.fData.end(), tSubPoints.fData.begin(), --(tSubPoints.fData.end()));
            continue;
        }
    }

    axialmeshmsg_debug("poly loop partitioned into <" << aPoints.fData.size() << "> closed points vertices" << eom);

    return;
}

//*********************
//tesselation functions
//*********************

void KGSimpleAxialMesher::OpenPointsToLoops(const OpenPoints& aPoints)
{
    OpenPoints::CIt tThisPoint;
    OpenPoints::CIt tNextPoint;

    //main hull cells
    tThisPoint = aPoints.fData.begin();
    tNextPoint = ++(aPoints.fData.begin());
    while (tNextPoint != aPoints.fData.end()) {
        Loop(*tThisPoint, *tNextPoint);

        ++tThisPoint;
        ++tNextPoint;
    }

    axialmeshmsg_debug("tesselated open points into <" << aPoints.fData.size() << "> loops" << eom);

    return;
}
void KGSimpleAxialMesher::ClosedPointsToLoops(const ClosedPoints& aPoints)
{
    OpenPoints::CIt tThisPoint;
    OpenPoints::CIt tNextPoint;

    //main hull cells
    tThisPoint = aPoints.fData.begin();
    tNextPoint = ++(aPoints.fData.begin());
    while (tNextPoint != aPoints.fData.end()) {
        Loop(*tThisPoint, *tNextPoint);

        ++tThisPoint;
        ++tNextPoint;
    }

    tThisPoint = --(aPoints.fData.end());
    tNextPoint = aPoints.fData.begin();
    Loop(*tThisPoint, *tNextPoint);

    axialmeshmsg_debug("tesselated closed points into <" << aPoints.fData.size() << "> loops" << eom);

    return;
}

//*************
//loop function
//*************

void KGSimpleAxialMesher::Loop(const KTwoVector& aFirst, const KTwoVector& aSecond)
{
    fCurrentElements->push_back(new KGAxialMeshLoop(aFirst, aSecond));

    return;
}

}  // namespace KGeoBag
