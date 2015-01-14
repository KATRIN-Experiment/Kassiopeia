#include "KGRotatedCircleSpaceAxialMesher.hh"

namespace KGeoBag
{

    KGRotatedCircleSpaceAxialMesher::KGRotatedCircleSpaceAxialMesher() :
            KGSimpleAxialMesher()
    {
    }
    KGRotatedCircleSpaceAxialMesher::~KGRotatedCircleSpaceAxialMesher()
    {
    }

    void KGRotatedCircleSpaceAxialMesher::VisitRotatedClosedPathSpace( KGRotatedCircleSpace* aRotatedCircleSpace )
    {
        //create poly line points
        ClosedPoints tCirclePoints;
        CircleToClosedPoints( aRotatedCircleSpace->Path().operator ->(), tCirclePoints );

        //create loops
        ClosedPointsToLoops( tCirclePoints );

        return;
    }

}
