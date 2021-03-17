#include "KGRotatedLineSegmentSpaceAxialMesher.hh"

namespace KGeoBag
{

KGRotatedLineSegmentSpaceAxialMesher::KGRotatedLineSegmentSpaceAxialMesher() = default;
KGRotatedLineSegmentSpaceAxialMesher::~KGRotatedLineSegmentSpaceAxialMesher() = default;

void KGRotatedLineSegmentSpaceAxialMesher::VisitRotatedOpenPathSpace(
    KGRotatedLineSegmentSpace* aRotatedLineSegmentSpace)
{
    //create line segment points
    OpenPoints tLineSegmentPoints;
    LineSegmentToOpenPoints(aRotatedLineSegmentSpace->Path().operator->(), tLineSegmentPoints);

    //create loops
    OpenPointsToLoops(tLineSegmentPoints);

    return;
}

}  // namespace KGeoBag
