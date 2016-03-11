//
// Created by trost on 03.06.15.
//

#ifndef KASPER_KSINTDECAYCALCULATORFERENCIONISATIONBUILDER_H
#define KASPER_KSINTDECAYCALCULATORFERENCIONISATIONBUILDER_H

#include "KComplexElement.hh"
#include "KSIntDecayCalculatorFerencIonisation.h"
#include "KSToolbox.h"


using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSIntDecayCalculatorFerencIonisation > KSIntDecayCalculatorFerencIonisationBuilder;

    template< >
    inline bool KSIntDecayCalculatorFerencIonisationBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "target_pid" )
        {
            aContainer->CopyTo( fObject, &KSIntDecayCalculatorFerencIonisation::SetTargetPID );
            return true;
        }
        if( aContainer->GetName() == "min_pid" )
        {
            aContainer->CopyTo( fObject, &KSIntDecayCalculatorFerencIonisation::SetminPID );
            return true;
        }
        if( aContainer->GetName() == "max_pid" )
        {
            aContainer->CopyTo( fObject, &KSIntDecayCalculatorFerencIonisation::SetminPID );
            return true;
        }
        if( aContainer->GetName() == "temperature" )
        {
            aContainer->CopyTo( fObject, &KSIntDecayCalculatorFerencIonisation::SetTemperature );
            return true;
        }

        return false;
    }

    template< >
    inline bool KSIntDecayCalculatorFerencIonisationBuilder::AddElement( KContainer* aContainer )
    {
        if( aContainer->Is< KSGenerator >() == true )
        {
            aContainer->ReleaseTo( fObject, &KSIntDecayCalculatorFerencIonisation::SetDecayProductGenerator );
            return true;
        }
        return false;
    }

}

#endif //KASPER_KSINTDECAYCALCULATORFERENCIONISATIONBUILDER_H
