#ifndef Kassiopeia_KSTermDeathBuilder_h_
#define Kassiopeia_KSTermDeathBuilder_h_

#include "KComplexElement.hh"
#include "KSTermDeath.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSTermDeath > KSTermDeathBuilder;

    template< >
    inline bool KSTermDeathBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        return false;
    }

}

#endif
