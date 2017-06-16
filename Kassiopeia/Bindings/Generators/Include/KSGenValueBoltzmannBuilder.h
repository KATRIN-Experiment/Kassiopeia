#ifndef Kassiopeia_KSGenValueBoltzmannBuilder_h_
#define Kassiopeia_KSGenValueBoltzmannBuilder_h_

#include "KComplexElement.hh"
#include "KSGenValueBoltzmann.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSGenValueBoltzmann > KSGenValueBoltzmannBuilder;

    template< >
    inline bool KSGenValueBoltzmannBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "value_mass" )
        {
            aContainer->CopyTo( fObject, &KSGenValueBoltzmann::SetValueMass );
            return true;
        }
        if( aContainer->GetName() == "value_kT" )
        {
            aContainer->CopyTo( fObject, &KSGenValueBoltzmann::SetValuekT );
            return true;
        }
        return false;
    }

}

#endif
