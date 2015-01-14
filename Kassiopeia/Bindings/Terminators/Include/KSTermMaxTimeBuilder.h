#ifndef Kassiopeia_KSTermMaxTimeBuilder_h_
#define Kassiopeia_KSTermMaxTimeBuilder_h_

#include "KComplexElement.hh"
#include "KSTermMaxTime.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSTermMaxTime > KSTermMaxTimeBuilder;

    template< >
    inline bool KSTermMaxTimeBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "time" )
        {
            aContainer->CopyTo( fObject, &KSTermMaxTime::SetTime );
            return true;
        }
        return false;
    }

}

#endif
