#ifndef Kassiopeia_KSFieldMagneticConstantBuilder_h_
#define Kassiopeia_KSFieldMagneticConstantBuilder_h_

#include "KComplexElement.hh"
#include "KSFieldMagneticConstant.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSFieldMagneticConstant > KSFieldMagneticConstantBuilder;

    template< >
    inline bool KSFieldMagneticConstantBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KSMagneticField::SetName );
            return true;
        }
        if( aContainer->GetName() == "field" )
        {
            aContainer->CopyTo( fObject, &KSFieldMagneticConstant::SetField );
            return true;
        }
        return false;
    }

}

#endif
