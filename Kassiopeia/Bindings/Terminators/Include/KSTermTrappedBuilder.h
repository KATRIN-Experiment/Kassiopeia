#ifndef Kassiopeia_KSTermTrappedBuilder_h_
#define Kassiopeia_KSTermTrappedBuilder_h_

#include "KComplexElement.hh"
#include "KSTermTrapped.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSTermTrapped > KSTermTrappedBuilder;

    template< >
    inline bool KSTermTrappedBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "max_turns" )
        {
            aContainer->CopyTo( fObject, &KSTermTrapped::SetMaxTurns );
            return true;
        }
        return false;
    }

}
#endif
