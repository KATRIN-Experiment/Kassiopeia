#ifndef Kassiopeia_KSRootSurfaceNavigatorBuilder_h_
#define Kassiopeia_KSRootSurfaceNavigatorBuilder_h_

#include "KComplexElement.hh"
#include "KSRootSurfaceNavigator.h"
#include "KSToolbox.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSRootSurfaceNavigator > KSRootSurfaceNavigatorBuilder;

    template< >
    inline bool KSRootSurfaceNavigatorBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "set_surface_navigator" )
        {
            fObject->SetSurfaceNavigator( KSToolbox::GetInstance()->GetObjectAs< KSSurfaceNavigator >( aContainer->AsReference< string >() ) );
            return true;
        }
        return false;
    }

}

#endif
