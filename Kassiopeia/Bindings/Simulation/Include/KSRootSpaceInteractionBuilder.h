#ifndef Kassiopeia_KSRootSpaceInteractionBuilder_h_
#define Kassiopeia_KSRootSpaceInteractionBuilder_h_

#include "KComplexElement.hh"
#include "KSRootSpaceInteraction.h"
#include "KSToolbox.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSRootSpaceInteraction > KSRootSpaceInteractionBuilder;

    template< >
    inline bool KSRootSpaceInteractionBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "add_space_interaction" )
        {
            fObject->AddSpaceInteraction( KSToolbox::GetInstance()->GetObjectAs< KSSpaceInteraction >( aContainer->AsReference< string >() ) );
            return true;
        }
        return false;
    }

}
#endif
