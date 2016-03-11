//
// Created by trost on 27.05.15.
//

#ifndef KASPER_KSINTDECAYCALCULATORFERENCSPONTANEOUSBUILDER_H
#define KASPER_KSINTDECAYCALCULATORFERENCSPONTANEOUSBUILDER_H

#include "KComplexElement.hh"
#include "KSIntDecayCalculatorFerencSpontaneous.h"
#include "KSToolbox.h"


using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSIntDecayCalculatorFerencSpontaneous > KSIntDecayCalculatorFerencSpontaneousBuilder;

    template< >
    inline bool KSIntDecayCalculatorFerencSpontaneousBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "target_pid" )
        {
            aContainer->CopyTo( fObject, &KSIntDecayCalculatorFerencSpontaneous::SetTargetPID );
            return true;
        }
        if( aContainer->GetName() == "min_pid" )
        {
            aContainer->CopyTo( fObject, &KSIntDecayCalculatorFerencSpontaneous::SetminPID );
            return true;
        }
        if( aContainer->GetName() == "max_pid" )
        {
            aContainer->CopyTo( fObject, &KSIntDecayCalculatorFerencSpontaneous::SetminPID );
            return true;
        }

        return false;
    }

}

#endif //KASPER_KSINTDECAYCALCULATORFERENCSPONTANEOUSBUILDER_H
