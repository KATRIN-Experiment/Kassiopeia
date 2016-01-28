#ifndef Kassiopeia_KSFieldMagneticRampedBuilder_h_
#define Kassiopeia_KSFieldMagneticRampedBuilder_h_

#include "KComplexElement.hh"
#include "KSFieldMagneticRamped.h"

using namespace Kassiopeia;
namespace katrin
{
    typedef KComplexElement< KSFieldMagneticRamped > KSFieldMagneticRampedBuilder;

    template< >
    inline bool KSFieldMagneticRampedBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KSMagneticField::SetName );
            return true;
        }
        if( aContainer->GetName() == "root_field" )
        {
            fObject->SetRootMagneticFieldName( aContainer->AsReference< string >() );
            return true;
        }
        if( aContainer->GetName() == "ramping_type" )
        {
            string tFlag = aContainer->AsReference< string >();
            if ( tFlag == string("linear") )
                fObject->SetRampingType( KSFieldMagneticRamped::rtLinear );
            else if ( tFlag == string("exponential") )
                fObject->SetRampingType( KSFieldMagneticRamped::rtExponential );
            else if ( tFlag == string("inversion") )
                fObject->SetRampingType( KSFieldMagneticRamped::rtInversion );
            else if ( tFlag == string("inversion2") )
                fObject->SetRampingType( KSFieldMagneticRamped::rtInversion2 );
            else if ( tFlag == string("flipbox") )
                fObject->SetRampingType( KSFieldMagneticRamped::rtFlipBox );
            return true;
        }
        if( aContainer->GetName() == "num_cycles" )
        {
            aContainer->CopyTo( fObject, &KSFieldMagneticRamped::SetNumCycles );
            return true;
        }
        if( aContainer->GetName() == "ramp_up_delay" )
        {
            aContainer->CopyTo( fObject, &KSFieldMagneticRamped::SetRampUpDelay );
            return true;
        }
        if( aContainer->GetName() == "ramp_down_delay" )
        {
            aContainer->CopyTo( fObject, &KSFieldMagneticRamped::SetRampDownDelay );
            return true;
        }
        if( aContainer->GetName() == "ramp_up_time" )
        {
            aContainer->CopyTo( fObject, &KSFieldMagneticRamped::SetRampUpTime );
            return true;
        }
        if( aContainer->GetName() == "ramp_down_time" )
        {
            aContainer->CopyTo( fObject, &KSFieldMagneticRamped::SetRampDownTime );
            return true;
        }
        if( aContainer->GetName() == "time_constant" )
        {
            aContainer->CopyTo( fObject, &KSFieldMagneticRamped::SetTimeConstant );
            return true;
        }
        if( aContainer->GetName() == "time_constant_2" )
        {
            aContainer->CopyTo( fObject, &KSFieldMagneticRamped::SetTimeConstant2 );
            return true;
        }
        if( aContainer->GetName() == "time_scaling" )
        {
            aContainer->CopyTo( fObject, &KSFieldMagneticRamped::SetTimeScalingFactor );
            return true;
        }
        return false;
    }

    /////////////////////////////////////////////////////////////////////////

    typedef KComplexElement< KSFieldElectricInducedAzimuthal > KSFieldElectricInducedAzimuthalBuilder;

    template< >
    inline bool KSFieldElectricInducedAzimuthalBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KSElectricField::SetName );
            return true;
        }
        if( aContainer->GetName() == "root_field" )
        {
            fObject->SetRampedMagneticFieldName( aContainer->AsReference< string >() );
            return true;
        }
        return false;
    }
}

#endif
