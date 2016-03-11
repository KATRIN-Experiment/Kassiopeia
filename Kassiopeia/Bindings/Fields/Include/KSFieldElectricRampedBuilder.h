#ifndef Kassiopeia_KSFieldElectricRampedBuilder_h_
#define Kassiopeia_KSFieldElectricRampedBuilder_h_

#include "KComplexElement.hh"
#include "KSFieldElectricRamped.h"

using namespace Kassiopeia;
namespace katrin
{
    typedef KComplexElement< KSFieldElectricRamped > KSFieldElectricRampedBuilder;

    template< >
    inline bool KSFieldElectricRampedBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KSElectricField::SetName );
            return true;
        }
        if( aContainer->GetName() == "root_field" )
        {
            fObject->SetRootElectricFieldName( aContainer->AsReference< string >() );
            return true;
        }
        if( aContainer->GetName() == "ramping_type" )
        {
            string tFlag = aContainer->AsReference< string >();
            if ( tFlag == string("linear") )
                fObject->SetRampingType( KSFieldElectricRamped::rtLinear );
            else if ( tFlag == string("exponential") )
                fObject->SetRampingType( KSFieldElectricRamped::rtExponential );
            return true;
        }
        if( aContainer->GetName() == "num_cycles" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectricRamped::SetNumCycles );
            return true;
        }
        if( aContainer->GetName() == "ramp_up_delay" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectricRamped::SetRampUpDelay );
            return true;
        }
        if( aContainer->GetName() == "ramp_down_delay" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectricRamped::SetRampDownDelay );
            return true;
        }
        if( aContainer->GetName() == "ramp_up_time" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectricRamped::SetRampUpTime );
            return true;
        }
        if( aContainer->GetName() == "ramp_down_time" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectricRamped::SetRampDownTime );
            return true;
        }
        if( aContainer->GetName() == "time_constant" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectricRamped::SetTimeConstant );
            return true;
        }
        if( aContainer->GetName() == "time_scaling" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectricRamped::SetTimeScalingFactor );
            return true;
        }
        return false;
    }

}

#endif
