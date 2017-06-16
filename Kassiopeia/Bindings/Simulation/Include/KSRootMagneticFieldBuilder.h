#ifndef Kassiopeia_KSRootMagneticFieldBuilder_h_
#define Kassiopeia_KSRootMagneticFieldBuilder_h_

#include "KComplexElement.hh"
#include "KSRootMagneticField.h"
#include "KToolbox.h"

#include "KSFieldFinder.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSRootMagneticField > KSRootMagneticFieldBuilder;

    template< >
    inline bool KSRootMagneticFieldBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "add_magnetic_field" )
        {
            fObject->AddMagneticField( getMagneticField( aContainer->AsReference< std::string >() ) );
            return true;
        }
        return false;
    }

}

#endif
