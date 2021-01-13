#include "KGRotatedPolyLoopSpaceMesher.hh"

namespace KGeoBag
{

KGRotatedPolyLoopSpaceMesher::KGRotatedPolyLoopSpaceMesher() = default;
KGRotatedPolyLoopSpaceMesher::~KGRotatedPolyLoopSpaceMesher() = default;

void KGRotatedPolyLoopSpaceMesher::VisitRotatedClosedPathSpace(KGRotatedPolyLoopSpace* aRotatedPolyLoopSpace)
{
    //create poly line points
    ClosedPoints tPolyLoopPoints;
    PolyLoopToClosedPoints(aRotatedPolyLoopSpace->Path().operator->(), tPolyLoopPoints);

    //create rotated points
    TorusMesh tMeshPoints;
    ClosedPointsRotatedToTorusMesh(tPolyLoopPoints, aRotatedPolyLoopSpace->RotatedMeshCount(), tMeshPoints);

    //create mesh
    TorusMeshToTriangles(tMeshPoints);

    return;
}

}  // namespace KGeoBag
