#ifndef Kassiopeia_KSRootRunModifierBuilder_h_
#define Kassiopeia_KSRootRunModifierBuilder_h_

#include "KComplexElement.hh"
#include "KSRootRunModifier.h"
#include "KToolbox.h"

using namespace Kassiopeia;
namespace katrin
{
    typedef KComplexElement< KSRootRunModifier > KSRootRunModifierBuilder;

    template< >
    inline bool KSRootRunModifierBuilder::AddAttribute(KContainer *aContainer)
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "add_modifier" )
        {
            fObject->AddModifier( KToolbox::GetInstance().Get<  KSRunModifier >( aContainer->AsReference< std::string >() ) );
            return true;
        }
        return false;
    }
}

#endif //Kassiopeia_KSRootRunModifierBuilder_h_
