//
// Created by trost on 29.05.15.
//

#ifndef KASPER_KSGENENERGYBETADECAYBUILDER_H
#define KASPER_KSGENENERGYBETADECAYBUILDER_H

#include "KComplexElement.hh"
#include "KSGenEnergyBetaDecay.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSGenEnergyBetaDecay > KSGenEnergyBetaDecayBuilder;

    template< >
    inline bool KSGenEnergyBetaDecayBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "daughter_z" )
        {
            aContainer->CopyTo( fObject, &KSGenEnergyBetaDecay::SetZDaughter );
            return true;
        }
        if( aContainer->GetName() == "nmax" )
        {
            aContainer->CopyTo( fObject, &KSGenEnergyBetaDecay::Setnmax );
            return true;
        }
        if( aContainer->GetName() == "endpoint_ev" )
        {
            aContainer->CopyTo( fObject, &KSGenEnergyBetaDecay::SetEndpoint );
            return true;
        }
        if( aContainer->GetName() == "mnu_ev" )
        {
            aContainer->CopyTo( fObject, &KSGenEnergyBetaDecay::Setmnu );
            return true;
        }
        if( aContainer->GetName() == "min_energy" )
        {
            aContainer->CopyTo( fObject, &KSGenEnergyBetaDecay::SetMinEnergy );
            return true;
        }
        if( aContainer->GetName() == "max_energy" )
        {
            aContainer->CopyTo( fObject, &KSGenEnergyBetaDecay::SetMaxEnergy );
            return true;
        }
        return false;
    }

}

#endif //KASPER_KSGENENERGYBETADECAYBUILDER_H
