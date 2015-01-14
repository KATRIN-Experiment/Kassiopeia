#ifndef Kassiopeia_KSGenTimeCompositeBuilder_h_
#define Kassiopeia_KSGenTimeCompositeBuilder_h_

#include "KComplexElement.hh"
#include "KSGenTimeComposite.h"
#include "KSToolbox.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSGenTimeComposite > KSGenTimeCompositeBuilder;

    template< >
    inline bool KSGenTimeCompositeBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "time_value" )
        {
            fObject->SetTimeValue( KSToolbox::GetInstance()->GetObjectAs< KSGenValue >( aContainer->AsReference< string >() ) );
            return true;
        }
        return false;
    }

    template< >
    inline bool KSGenTimeCompositeBuilder::AddElement( KContainer* aContainer )
    {
        if( aContainer->GetName().substr( 0, 4 ) == "time" )
        {
            aContainer->ReleaseTo( fObject, &KSGenTimeComposite::SetTimeValue );
            return true;
        }
        return false;
    }

}
#endif
