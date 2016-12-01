#ifndef Kassiopeia_KSRootElectricFieldBuilder_h_
#define Kassiopeia_KSRootElectricFieldBuilder_h_

#include "KComplexElement.hh"
#include "KSRootElectricField.h"
#include "KToolbox.h"

#include "KSFieldFinder.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSRootElectricField > KSRootElectricFieldBuilder;

    template< >
    inline bool KSRootElectricFieldBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "add_electric_field" )
        {
            fObject->AddElectricField( getElectricField( aContainer->AsReference< std::string >() ) );
            return true;
        }
        return false;
    }

}
#endif
