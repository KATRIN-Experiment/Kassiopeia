#ifndef Kassiopeia_KSFieldMagneticDipoleBuilder_h_
#define Kassiopeia_KSFieldMagneticDipoleBuilder_h_

#include "KComplexElement.hh"
#include "KSFieldMagneticDipole.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSFieldMagneticDipole > KSFieldMagneticDipoleBuilder;

    template< >
    inline bool KSFieldMagneticDipoleBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KSMagneticField::SetName );
            return true;
        }
        if( aContainer->GetName() == "location" )
        {
            aContainer->CopyTo( fObject, &KSFieldMagneticDipole::SetLocation );
            return true;
        }
        if( aContainer->GetName() == "moment" )
        {
            aContainer->CopyTo( fObject, &KSFieldMagneticDipole::SetMoment );
            return true;
        }
        return false;
    }

}

#endif
