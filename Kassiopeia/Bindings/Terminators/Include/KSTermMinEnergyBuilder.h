#ifndef Kassiopeia_KSTermMinEnergyBuilder_h_
#define Kassiopeia_KSTermMinEnergyBuilder_h_

#include "KComplexElement.hh"
#include "KSTermMinEnergy.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSTermMinEnergy > KSTermMinEnergyBuilder;

    template< >
    inline bool KSTermMinEnergyBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "energy" )
        {
            aContainer->CopyTo( fObject, &KSTermMinEnergy::SetMinEnergy );
            return true;
        }
        return false;
    }

}

#endif
