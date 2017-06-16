#ifndef Kassiopeia_KSGenValueAngleCosineBuilder_h_
#define Kassiopeia_KSGenValueAngleCosineBuilder_h_

#include "KComplexElement.hh"
#include "KSGenValueAngleCosine.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSGenValueAngleCosine > KSGenValueAngleCosineBuilder;

    template< >
    inline bool KSGenValueAngleCosineBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "angle_min" )
        {
            aContainer->CopyTo( fObject, &KSGenValueAngleCosine::SetAngleMin );
            return true;
        }
        if( aContainer->GetName() == "angle_max" )
        {
            aContainer->CopyTo( fObject, &KSGenValueAngleCosine::SetAngleMax );
            return true;
        }
        return false;
    }

}

#endif
