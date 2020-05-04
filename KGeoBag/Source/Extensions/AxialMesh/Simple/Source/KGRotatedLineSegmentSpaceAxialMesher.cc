#include "KGRotatedLineSegmentSpaceAxialMesher.hh"

namespace KGeoBag
{

KGRotatedLineSegmentSpaceAxialMesher::KGRotatedLineSegmentSpaceAxialMesher() : KGSimpleAxialMesher() {}
KGRotatedLineSegmentSpaceAxialMesher::~KGRotatedLineSegmentSpaceAxialMesher() {}

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
