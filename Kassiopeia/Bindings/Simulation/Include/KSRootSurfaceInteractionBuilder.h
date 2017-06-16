#ifndef Kassiopeia_KSRootSurfaceInteractionBuilder_h_
#define Kassiopeia_KSRootSurfaceInteractionBuilder_h_

#include "KComplexElement.hh"
#include "KSRootSurfaceInteraction.h"
#include "KToolbox.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSRootSurfaceInteraction > KSRootSurfaceInteractionBuilder;

    template< >
    inline bool KSRootSurfaceInteractionBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "set_surface_interaction" )
        {
            fObject->SetSurfaceInteraction( KToolbox::GetInstance().Get< KSSurfaceInteraction >( aContainer->AsReference< std::string >() ) );
            return true;
        }
        return false;
    }

}
#endif
