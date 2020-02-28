//
// Created by wdconinc on 13.02.20.
//

#ifndef KASPER_KSGENENERGYBETARECOILBUILDER_H
#define KASPER_KSGENENERGYBETARECOILBUILDER_H

#include "KComplexElement.hh"
#include "KSGenEnergyBetaRecoil.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSGenEnergyBetaRecoil > KSGenEnergyBetaRecoilBuilder;

    template< >
    inline bool KSGenEnergyBetaRecoilBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "nmax" )
        {
            aContainer->CopyTo( fObject, &KSGenEnergyBetaRecoil::SetNMax );
            return true;
        }
        if( aContainer->GetName() == "min_energy" )
        {
            aContainer->CopyTo( fObject, &KSGenEnergyBetaRecoil::SetMinEnergy );
            return true;
        }
        if( aContainer->GetName() == "max_energy" )
        {
            aContainer->CopyTo( fObject, &KSGenEnergyBetaRecoil::SetMaxEnergy );
            return true;
        }
        return false;
    }

}

#endif //KASPER_KSGENENERGYBETARECOILBUILDER_H
