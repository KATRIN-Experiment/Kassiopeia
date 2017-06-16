#ifndef Kassiopeia_KSIntDecayCalculatorGlukhovDeExcitationBuilder_h_
#define Kassiopeia_KSIntDecayCalculatorGlukhovDeExcitationBuilder_h_

#include "KComplexElement.hh"
#include "KSIntDecayCalculatorGlukhovDeExcitation.h"
#include "KToolbox.h"


using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSIntDecayCalculatorGlukhovDeExcitation > KSIntDecayCalculatorGlukhovDeExcitationBuilder;

    template< >
    inline bool KSIntDecayCalculatorGlukhovDeExcitationBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "target_pid" )
        {
            aContainer->CopyTo( fObject, &KSIntDecayCalculatorGlukhovDeExcitation::SetTargetPID );
            return true;
        }
        if( aContainer->GetName() == "min_pid" )
        {
            aContainer->CopyTo( fObject, &KSIntDecayCalculatorGlukhovDeExcitation::SetminPID );
            return true;
        }
        if( aContainer->GetName() == "max_pid" )
        {
            aContainer->CopyTo( fObject, &KSIntDecayCalculatorGlukhovDeExcitation::SetminPID );
            return true;
        }
        if( aContainer->GetName() == "temperature" )
        {
            aContainer->CopyTo( fObject, &KSIntDecayCalculatorGlukhovDeExcitation::SetTemperature );
            return true;
        }

        return false;
    }

}

#endif
