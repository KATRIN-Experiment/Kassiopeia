#ifndef Kassiopeia_KSFieldElectricConstantBuilder_h_
#define Kassiopeia_KSFieldElectricConstantBuilder_h_

#include "KSFieldElectricConstant.h"
#include "KComplexElement.hh"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSFieldElectricConstant > KSFieldElectricConstantBuilder;

    template< >
    inline bool KSFieldElectricConstantBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectricConstant::SetName );
            return true;
        }
        if( aContainer->GetName() == "field" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectricConstant::SetField );
            return true;
        }
        return false;
    }

}

#endif
