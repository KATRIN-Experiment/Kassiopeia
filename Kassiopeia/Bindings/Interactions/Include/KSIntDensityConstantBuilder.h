#ifndef Kassiopeia_KSIntDensityConstantBuilder_h_
#define Kassiopeia_KSIntDensityConstantBuilder_h_

#include "KComplexElement.hh"
#include "KSIntDensityConstant.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSIntDensityConstant > KSIntDensityConstantBuilder;

    template< >
    inline bool KSIntDensityConstantBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "temperature" )
        {
            aContainer->CopyTo( fObject, &KSIntDensityConstant::SetTemperature );
            return true;
        }
        if( aContainer->GetName() == "pressure" )
        {
            aContainer->CopyTo( fObject, &KSIntDensityConstant::SetPressure );
            return true;
        }
        if( aContainer->GetName() == "pressure_mbar" )
        {
            double aPascalPressure = aContainer->AsReference<double>() * 100;
            fObject->SetPressure( aPascalPressure );
            return true;
        }
        if( aContainer->GetName() == "density" )
        {
            aContainer->CopyTo( fObject, &KSIntDensityConstant::SetDensity );
            return true;
        }
        return false;
    }

}

#endif
