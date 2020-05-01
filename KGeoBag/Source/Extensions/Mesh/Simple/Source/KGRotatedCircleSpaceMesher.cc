#include "KGRotatedCircleSpaceMesher.hh"

namespace KGeoBag
{

KGRotatedCircleSpaceMesher::KGRotatedCircleSpaceMesher() : KGSimpleMesher() {}
KGRotatedCircleSpaceMesher::~KGRotatedCircleSpaceMesher() {}

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
