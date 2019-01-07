#ifndef Kassiopeia_KSIntSpinRotateYPulseBuilder_h_
#define Kassiopeia_KSIntSpinRotateYPulseBuilder_h_

#include "KComplexElement.hh"
#include "KSIntSpinRotateYPulse.h"
#include "KToolbox.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSIntSpinRotateYPulse > KSIntSpinRotateYPulseBuilder;

    template< >
    inline bool KSIntSpinRotateYPulseBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "time" )
        {
            aContainer->CopyTo( fObject, &KSIntSpinRotateYPulse::SetTime );
            return true;
        }
        if( aContainer->GetName() == "angle" )
        {
            aContainer->CopyTo( fObject, &KSIntSpinRotateYPulse::SetAngle );
            return true;
        }
        if( aContainer->GetName() == "is_adiabatic" )
        {
            aContainer->CopyTo( fObject, &KSIntSpinRotateYPulse::SetIsAdiabatic );
            return true;
        }
        return false;
    }

}

#endif
