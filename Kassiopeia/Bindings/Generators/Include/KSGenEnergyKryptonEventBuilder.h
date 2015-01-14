#ifndef Kassiopeia_KSGenEnergyKryptonEventBuilder_h_
#define Kassiopeia_KSGenEnergyKryptonEventBuilder_h_

#include "KComplexElement.hh"
#include "KSGenEnergyKryptonEvent.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSGenEnergyKryptonEvent > KSGenEnergyKryptonEventBuilder;

    template< >
    inline bool KSGenEnergyKryptonEventBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "force_conversion" )
        {
            aContainer->CopyTo( fObject, &KSGenEnergyKryptonEvent::SetForceConversion );
            return true;
        }
        if( aContainer->GetName() == "do_conversion" )
        {
            aContainer->CopyTo( fObject, &KSGenEnergyKryptonEvent::SetDoConversion );
            return true;
        }
        if( aContainer->GetName() == "do_auger" )
        {
            aContainer->CopyTo( fObject, &KSGenEnergyKryptonEvent::SetDoAuger );
            return true;
        }
        return false;
    }

}
#endif
