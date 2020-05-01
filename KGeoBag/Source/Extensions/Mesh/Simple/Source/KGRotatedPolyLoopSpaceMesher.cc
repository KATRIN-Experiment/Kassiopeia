#include "KGRotatedPolyLoopSpaceMesher.hh"

namespace KGeoBag
{

KGRotatedPolyLoopSpaceMesher::KGRotatedPolyLoopSpaceMesher() : KGSimpleMesher() {}
KGRotatedPolyLoopSpaceMesher::~KGRotatedPolyLoopSpaceMesher() {}

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
