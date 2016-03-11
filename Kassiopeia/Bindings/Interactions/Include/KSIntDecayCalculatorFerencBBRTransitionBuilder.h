//
// Created by trost on 27.05.15.
//

#ifndef KASPER_KSINTDECAYCALCULATORFERENCBBRTRANSITIONBUILDER_H
#define KASPER_KSINTDECAYCALCULATORFERENCBBRTRANSITIONBUILDER_H

#include "KComplexElement.hh"
#include "KSIntDecayCalculatorFerencBBRTransition.h"
#include "KSToolbox.h"


using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSIntDecayCalculatorFerencBBRTransition > KSIntDecayCalculatorFerencBBRTransitionBuilder;

    template< >
    inline bool KSIntDecayCalculatorFerencBBRTransitionBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "target_pid" )
        {
            aContainer->CopyTo( fObject, &KSIntDecayCalculatorFerencBBRTransition::SetTargetPID );
            return true;
        }
        if( aContainer->GetName() == "min_pid" )
        {
            aContainer->CopyTo( fObject, &KSIntDecayCalculatorFerencBBRTransition::SetminPID );
            return true;
        }
        if( aContainer->GetName() == "max_pid" )
        {
            aContainer->CopyTo( fObject, &KSIntDecayCalculatorFerencBBRTransition::SetminPID );
            return true;
        }
        if( aContainer->GetName() == "temperature" )
        {
            aContainer->CopyTo( fObject, &KSIntDecayCalculatorFerencBBRTransition::SetTemperature );
            return true;
        }

        return false;
    }

}

#endif //KASPER_KSINTDECAYCALCULATORFERENCBBRTRANSITIONBUILDER_H
