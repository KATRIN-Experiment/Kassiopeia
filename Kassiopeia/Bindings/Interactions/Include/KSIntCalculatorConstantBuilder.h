#ifndef Kassiopeia_KSIntCalculatorConstantBuilder_h_
#define Kassiopeia_KSIntCalculatorConstantBuilder_h_

#include "KComplexElement.hh"
#include "KSIntCalculatorConstant.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSIntCalculatorConstant > KSIntCalculatorConstantBuilder;

    template< >
    inline bool KSIntCalculatorConstantBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "cross_section" )
        {
            aContainer->CopyTo( fObject, &KSIntCalculatorConstant::SetCrossSection );
            return true;
        }
        return false;
    }

}

#endif
