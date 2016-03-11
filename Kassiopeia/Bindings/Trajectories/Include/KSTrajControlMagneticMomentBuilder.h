#ifndef Kassiopeia_KSTrajControlMagneticMomentBuilder_h_
#define Kassiopeia_KSTrajControlMagneticMomentBuilder_h_

#include "KComplexElement.hh"
#include "KSTrajControlMagneticMoment.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSTrajControlMagneticMoment > KSTrajControlMagneticMomentBuilder;

    template< >
    inline bool KSTrajControlMagneticMomentBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "lower_limit" )
        {
            aContainer->CopyTo( fObject, &KSTrajControlMagneticMoment::SetLowerLimit );
            return true;
        }
        if( aContainer->GetName() == "upper_limit" )
        {
            aContainer->CopyTo( fObject, &KSTrajControlMagneticMoment::SetUpperLimit );
            return true;
        }
        return false;
    }

}
#endif
