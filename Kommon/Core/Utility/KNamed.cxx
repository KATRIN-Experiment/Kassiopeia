#include "KNamed.h"

namespace katrin
{

    KNamed::KNamed() :
        fName( "(anonymous)" )
    {
    }
    KNamed::KNamed( const KNamed& aNamed ) :
        fName( aNamed.fName )
    {
    }
    KNamed::~KNamed()
    {
    }

}
