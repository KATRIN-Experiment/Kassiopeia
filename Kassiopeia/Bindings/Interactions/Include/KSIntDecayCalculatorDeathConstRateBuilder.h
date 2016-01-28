#ifndef Kassiopeia_KSIntDecayCalculatorDeathConstRateBuilder_h_
#define Kassiopeia_KSIntDecayCalculatorDeathConstRateBuilder_h_

#include "KComplexElement.hh"
#include "KSIntDecayCalculatorDeathConstRate.h"
#include "KSToolbox.h"


using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSIntDecayCalculatorDeathConstRate > KSIntDecayCalculatorDeathConstRateBuilder;

    template< >
    inline bool KSIntDecayCalculatorDeathConstRateBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "life_time" )
        {
            aContainer->CopyTo( fObject, &KSIntDecayCalculatorDeathConstRate::SetLifeTime );
            return true;
        }
        if( aContainer->GetName() == "target_pid" )
        {
            aContainer->CopyTo( fObject, &KSIntDecayCalculatorDeathConstRate::SetTargetPID );
            return true;
        }
        if( aContainer->GetName() == "min_pid" )
        {
            aContainer->CopyTo( fObject, &KSIntDecayCalculatorDeathConstRate::SetminPID );
            return true;
        }
        if( aContainer->GetName() == "max_pid" )
        {
            aContainer->CopyTo( fObject, &KSIntDecayCalculatorDeathConstRate::SetminPID );
            return true;
        }

        return false;
    }

    template< >
    inline bool KSIntDecayCalculatorDeathConstRateBuilder::AddElement( KContainer* aContainer )
    {
        if( aContainer->Is< KSGenerator >() == true )
        {
            aContainer->ReleaseTo( fObject, &KSIntDecayCalculatorDeathConstRate::SetDecayProductGenerator );
            return true;
        }
        return false;
    }

}

#endif
