#include "KSRootStepModifierBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{
    template< >
    KSRootStepModifierBuilder::~KComplexElement()
    {
    }

    STATICINT sKSRootStepModifier =
            KSRootBuilder::ComplexElement< KSRootStepModifier >( "ks_root_stepmodifier" );

    STATICINT sKSRootStepModifierStructure =
            KSRootStepModifierBuilder::Attribute< string >( "name" ) +
            KSRootStepModifierBuilder::Attribute< string >( "add_modifier" );
}
