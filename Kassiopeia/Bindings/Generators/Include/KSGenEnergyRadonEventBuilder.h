#ifndef Kassiopeia_KSGenEnergyRadonEventBuilder_h_
#define Kassiopeia_KSGenEnergyRadonEventBuilder_h_

#include "KComplexElement.hh"
#include "KSGenEnergyRadonEvent.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSGenEnergyRadonEvent > KSGenEnergyRadonEventBuilder;

    template< >
    inline bool KSGenEnergyRadonEventBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "force_shake_off" )
        {
            aContainer->CopyTo( fObject, &KSGenEnergyRadonEvent::SetForceShakeOff );
            return true;
        }
        if( aContainer->GetName() == "force_conversion" )
        {
            aContainer->CopyTo( fObject, &KSGenEnergyRadonEvent::SetForceConversion );
            return true;
        }
        if( aContainer->GetName() == "do_shake_off" )
        {
            aContainer->CopyTo( fObject, &KSGenEnergyRadonEvent::SetDoShakeOff );
            return true;
        }
        if( aContainer->GetName() == "do_conversion" )
        {
            aContainer->CopyTo( fObject, &KSGenEnergyRadonEvent::SetDoConversion );
            return true;
        }
        if( aContainer->GetName() == "do_auger" )
        {
            aContainer->CopyTo( fObject, &KSGenEnergyRadonEvent::SetDoAuger );
            return true;
        }
        if( aContainer->GetName() == "isotope_number" )
        {
            aContainer->CopyTo( fObject, &KSGenEnergyRadonEvent::SetIsotope );
            return true;
        }
        return false;
    }

}

#endif
