#ifndef Kassiopeia_KSRootTrackModifierBuilder_h_
#define Kassiopeia_KSRootTrackModifierBuilder_h_

#include "KComplexElement.hh"
#include "KSRootTrackModifier.h"
#include "KToolbox.h"

using namespace Kassiopeia;
namespace katrin
{
    typedef KComplexElement< KSRootTrackModifier > KSRootTrackModifierBuilder;

    template< >
    inline bool KSRootTrackModifierBuilder::AddAttribute(KContainer *aContainer)
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "add_modifier" )
        {
            fObject->AddModifier( KToolbox::GetInstance().Get< KSTrackModifier >( aContainer->AsReference< std::string >() ) );
            return true;
        }
        return false;
    }
}

#endif //Kassiopeia_KSRootTrackModifierBuilder_h_
