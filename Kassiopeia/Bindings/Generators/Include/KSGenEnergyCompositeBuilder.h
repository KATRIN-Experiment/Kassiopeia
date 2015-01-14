#ifndef Kassiopeia_KSGenEnergyCompositeBuilder_h_
#define Kassiopeia_KSGenEnergyCompositeBuilder_h_

#include "KComplexElement.hh"
#include "KSGenEnergyComposite.h"
#include "KSToolbox.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSGenEnergyComposite > KSGenEnergyCompositeBuilder;

    template< >
    inline bool KSGenEnergyCompositeBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "energy" )
        {
            fObject->SetEnergyValue( KSToolbox::GetInstance()->GetObjectAs< KSGenValue >( aContainer->AsReference< string >() ) );
            return true;
        }
        return false;
    }

    template< >
    inline bool KSGenEnergyCompositeBuilder::AddElement( KContainer* aContainer )
    {
        if( aContainer->GetName().substr( 0, 6 ) == "energy" )
        {
            aContainer->ReleaseTo( fObject, &KSGenEnergyComposite::SetEnergyValue );
            return true;
        }
        return false;
    }

}

#endif
