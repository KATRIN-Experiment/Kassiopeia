//
// Created by trost on 14.03.16.
//

#ifndef KASPER_KSGENENERGYRYDBERGBUILDER_H
#define KASPER_KSGENENERGYRYDBERGBUILDER_H

#include "KComplexElement.hh"
#include "KSGenEnergyRydberg.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement< KSGenEnergyRydberg > KSGenEnergyRydbergBuilder;

template< >
inline bool KSGenEnergyRydbergBuilder::AddAttribute( KContainer* aContainer )
{
    if( aContainer->GetName() == "name" )
    {
        aContainer->CopyTo( fObject, &KNamed::SetName );
        return true;
    }
    if( aContainer->GetName() == "ionization_energy" )
    {
        aContainer->CopyTo( fObject, &KSGenEnergyRydberg::SetIonizationEnergy );
        return true;
    }
    if( aContainer->GetName() == "deposited_energy" )
    {
        aContainer->CopyTo( fObject, &KSGenEnergyRydberg::SetDepositedEnergy );
        return true;
    }

    return false;
}

}

#endif //KASPER_KSGENENERGYRYDBERGBUILDER_H
