#include "KGRotatedPolyLineSpaceAxialMesher.hh"

namespace KGeoBag
{

KGRotatedPolyLineSpaceAxialMesher::KGRotatedPolyLineSpaceAxialMesher() = default;
KGRotatedPolyLineSpaceAxialMesher::~KGRotatedPolyLineSpaceAxialMesher() = default;

void KGRotatedPolyLineSpaceAxialMesher::VisitRotatedOpenPathSpace(KGRotatedPolyLineSpace* aRotatedPolyLineSpace)
{
    //create line segment points
    OpenPoints tPolyLinePoints;
    PolyLineToOpenPoints(aRotatedPolyLineSpace->Path().operator->(), tPolyLinePoints);

    //create loops
    OpenPointsToLoops(tPolyLinePoints);

    return;
}

}  // namespace KGeoBag
