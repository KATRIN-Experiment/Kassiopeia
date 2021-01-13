#include "KGRotatedPolyLoopSpaceAxialMesher.hh"

namespace KGeoBag
{

KGRotatedPolyLoopSpaceAxialMesher::KGRotatedPolyLoopSpaceAxialMesher() = default;
KGRotatedPolyLoopSpaceAxialMesher::~KGRotatedPolyLoopSpaceAxialMesher() = default;

void KGRotatedPolyLoopSpaceAxialMesher::VisitRotatedClosedPathSpace(KGRotatedPolyLoopSpace* aRotatedPolyLoopSpace)
{
    //create poly line points
    ClosedPoints tPolyLoopPoints;
    PolyLoopToClosedPoints(aRotatedPolyLoopSpace->Path().operator->(), tPolyLoopPoints);

    //create loops
    ClosedPointsToLoops(tPolyLoopPoints);

    return;
}

}  // namespace KGeoBag
