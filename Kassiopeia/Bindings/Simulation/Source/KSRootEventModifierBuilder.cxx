#include "KSRootEventModifierBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{
    template< >
    KSRootEventModifierBuilder::~KComplexElement()
    {
    }

    STATICINT sKSRootEventModifier =
            KSRootBuilder::ComplexElement< KSRootEventModifier >( "ks_root_eventmodifier" );

    STATICINT sKSRootEventModifierStructure =
            KSRootEventModifierBuilder::Attribute< string >( "name" ) +
            KSRootEventModifierBuilder::Attribute< string >( "add_modifier" );
}
