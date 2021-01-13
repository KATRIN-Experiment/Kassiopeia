#include "KGRotatedCircleSpaceMesher.hh"

namespace KGeoBag
{

KGRotatedCircleSpaceMesher::KGRotatedCircleSpaceMesher() = default;
KGRotatedCircleSpaceMesher::~KGRotatedCircleSpaceMesher() = default;

void KGRotatedCircleSpaceMesher::VisitRotatedClosedPathSpace(KGRotatedCircleSpace* aRotatedCircleSpace)
{
    //create poly line points
    ClosedPoints tCirclePoints;
    CircleToClosedPoints(aRotatedCircleSpace->Path().operator->(), tCirclePoints);

    //create rotated points
    TorusMesh tMeshPoints;
    ClosedPointsRotatedToTorusMesh(tCirclePoints, aRotatedCircleSpace->RotatedMeshCount(), tMeshPoints);

    //create mesh
    TorusMeshToTriangles(tMeshPoints);

    return;
}

}  // namespace KGeoBag
