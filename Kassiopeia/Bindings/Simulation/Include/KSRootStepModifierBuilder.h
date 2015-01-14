#ifndef Kassiopeia_KSRootStepModifierBuilder_h_
#define Kassiopeia_KSRootStepModifierBuilder_h_

#include "KComplexElement.hh"
#include "KSRootStepModifier.h"
#include "KSToolbox.h"

using namespace Kassiopeia;
namespace katrin
{
    typedef KComplexElement< KSRootStepModifier > KSRootStepModifierBuilder;

    template< >
    inline bool KSRootStepModifierBuilder::AddAttribute(KContainer *aContainer)
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "add_modifier" )
        {
            fObject->AddModifier( KSToolbox::GetInstance()->GetObjectAs< KSStepModifier >( aContainer->AsReference< string >() ) );
            return true;
        }
        return false;
    }
}

#endif //Kassiopeia_KSRootStepModifierBuilder_h_