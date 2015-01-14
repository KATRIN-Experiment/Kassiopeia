#include "KGRotatedPolyLineSpaceAxialMesher.hh"

namespace KGeoBag
{

    KGRotatedPolyLineSpaceAxialMesher::KGRotatedPolyLineSpaceAxialMesher() :
            KGSimpleAxialMesher()
    {
    }
    KGRotatedPolyLineSpaceAxialMesher::~KGRotatedPolyLineSpaceAxialMesher()
    {
    }

    void KGRotatedPolyLineSpaceAxialMesher::VisitRotatedOpenPathSpace( KGRotatedPolyLineSpace* aRotatedPolyLineSpace )
    {
        //create line segment points
        OpenPoints tPolyLinePoints;
        PolyLineToOpenPoints( aRotatedPolyLineSpace->Path().operator ->(), tPolyLinePoints );

        //create loops
        OpenPointsToLoops( tPolyLinePoints );

        return;
    }

}
