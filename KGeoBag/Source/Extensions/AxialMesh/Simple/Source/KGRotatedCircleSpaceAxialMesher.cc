#include "KGRotatedCircleSpaceAxialMesher.hh"

namespace KGeoBag
{

KGRotatedCircleSpaceAxialMesher::KGRotatedCircleSpaceAxialMesher() = default;
KGRotatedCircleSpaceAxialMesher::~KGRotatedCircleSpaceAxialMesher() = default;

void KGRotatedCircleSpaceAxialMesher::VisitRotatedClosedPathSpace(KGRotatedCircleSpace* aRotatedCircleSpace)
{
    //create poly line points
    ClosedPoints tCirclePoints;
    CircleToClosedPoints(aRotatedCircleSpace->Path().operator->(), tCirclePoints);

    //create loops
    ClosedPointsToLoops(tCirclePoints);

    return;
}

}  // namespace KGeoBag
