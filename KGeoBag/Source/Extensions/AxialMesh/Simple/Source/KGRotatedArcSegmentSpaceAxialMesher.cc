#include "KGRotatedArcSegmentSpaceAxialMesher.hh"

namespace KGeoBag
{

    KGRotatedArcSegmentSpaceAxialMesher::KGRotatedArcSegmentSpaceAxialMesher() :
            KGSimpleAxialMesher()
    {
    }
    KGRotatedArcSegmentSpaceAxialMesher::~KGRotatedArcSegmentSpaceAxialMesher()
    {
    }

    void KGRotatedArcSegmentSpaceAxialMesher::VisitRotatedOpenPathSpace( KGRotatedArcSegmentSpace* aRotatedArcSegmentSpace )
    {
        //create line segment points
        OpenPoints tArcSegmentPoints;
        ArcSegmentToOpenPoints( aRotatedArcSegmentSpace->Path().operator ->(), tArcSegmentPoints );

        //create loops
        OpenPointsToLoops( tArcSegmentPoints );

        return;
    }

}
