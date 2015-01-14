#ifndef Kassiopeia_KSTermMinLongEnergyBuilder_h_
#define Kassiopeia_KSTermMinLongEnergyBuilder_h_

#include "KComplexElement.hh"
#include "KSTermMinLongEnergy.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSTermMinLongEnergy > KSTermMinLongEnergyBuilder;

    template< >
    inline bool KSTermMinLongEnergyBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "long_energy" )
        {
            aContainer->CopyTo( fObject, &KSTermMinLongEnergy::SetMinLongEnergy );
            return true;
        }
        return false;
    }

}

#endif
