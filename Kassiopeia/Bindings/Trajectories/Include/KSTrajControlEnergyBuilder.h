#ifndef Kassiopeia_KSTrajControlEnergyBuilder_h_
#define Kassiopeia_KSTrajControlEnergyBuilder_h_

#include "KComplexElement.hh"
#include "KSTrajControlEnergy.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSTrajControlEnergy > KSTrajControlEnergyBuilder;

    template< >
    inline bool KSTrajControlEnergyBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "lower_limit" )
        {
            aContainer->CopyTo( fObject, &KSTrajControlEnergy::SetLowerLimit );
            return true;
        }
        if( aContainer->GetName() == "upper_limit" )
        {
            aContainer->CopyTo( fObject, &KSTrajControlEnergy::SetUpperLimit );
            return true;
        }
        return false;
    }

}
#endif
