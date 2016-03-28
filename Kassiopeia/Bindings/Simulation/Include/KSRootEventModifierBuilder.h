#ifndef Kassiopeia_KSRootEventModifierBuilder_h_
#define Kassiopeia_KSRootEventModifierBuilder_h_

#include "KComplexElement.hh"
#include "KSRootEventModifier.h"
#include "KSToolbox.h"

using namespace Kassiopeia;
namespace katrin
{
    typedef KComplexElement< KSRootEventModifier > KSRootEventModifierBuilder;

    template< >
    inline bool KSRootEventModifierBuilder::AddAttribute(KContainer *aContainer)
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "add_modifier" )
        {
            fObject->AddModifier( KSToolbox::GetInstance()->GetObjectAs< KSEventModifier >( aContainer->AsReference< string >() ) );
            return true;
        }
        return false;
    }
}

#endif //Kassiopeia_KSRootEventModifierBuilder_h_
