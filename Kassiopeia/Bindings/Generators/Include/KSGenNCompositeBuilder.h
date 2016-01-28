#ifndef Kassiopeia_KSGenNCompositeBuilder_h_
#define Kassiopeia_KSGenNCompositeBuilder_h_

#include "KComplexElement.hh"
#include "KSGenNComposite.h"
#include "KSToolbox.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSGenNComposite > KSGenNCompositeBuilder;

    template< >
    inline bool KSGenNCompositeBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "n_value" )
        {
            fObject->SetNValue( KSToolbox::GetInstance()->GetObjectAs< KSGenValue >( aContainer->AsReference< string >() ) );
            return true;
        }
        return false;
    }

    template< >
    inline bool KSGenNCompositeBuilder::AddElement( KContainer* aContainer )
    {
        if( aContainer->GetName().substr( 0, 1 ) == "n" )
        {
            aContainer->ReleaseTo( fObject, &KSGenNComposite::SetNValue );
            return true;
        }
        return false;
    }

}
#endif
