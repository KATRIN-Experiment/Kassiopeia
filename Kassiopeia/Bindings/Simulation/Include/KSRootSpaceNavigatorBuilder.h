#ifndef Kassiopeia_KSRootSpaceNavigatorBuilder_h_
#define Kassiopeia_KSRootSpaceNavigatorBuilder_h_

#include "KComplexElement.hh"
#include "KSRootSpaceNavigator.h"
#include "KSToolbox.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSRootSpaceNavigator > KSRootSpaceNavigatorBuilder;

    template< >
    inline bool KSRootSpaceNavigatorBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "set_space_navigator" )
        {
            fObject->SetSpaceNavigator( KSToolbox::GetInstance()->GetObjectAs< KSSpaceNavigator >( aContainer->AsReference< string >() ) );
            return true;
        }
        return false;
    }

}

#endif
