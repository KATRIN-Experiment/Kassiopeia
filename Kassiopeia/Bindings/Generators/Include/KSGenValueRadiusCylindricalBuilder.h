#ifndef Kassiopeia_KSGenValueRadiusCylindricalBuilder_h_
#define Kassiopeia_KSGenValueRadiusCylindricalBuilder_h_

#include "KComplexElement.hh"
#include "KSGenValueRadiusCylindrical.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSGenValueRadiusCylindrical > KSGenValueRadiusCylindricalBuilder;

    template< >
    inline bool KSGenValueRadiusCylindricalBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "radius_min" )
        {
            aContainer->CopyTo( fObject, &KSGenValueRadiusCylindrical::SetRadiusMin );
            return true;
        }
        if( aContainer->GetName() == "radius_max" )
        {
            aContainer->CopyTo( fObject, &KSGenValueRadiusCylindrical::SetRadiusMax );
            return true;
        }
        return false;
    }

}

#endif
