#include "KSRootRunModifierBuilder.h"
#include "KSRootBuilder.h"
#include <string>

using namespace Kassiopeia;
namespace katrin
{
    template< >
    KSRootRunModifierBuilder::~KComplexElement()
    {
    }

    STATICINT sKSRootRunModifier =
            KSRootBuilder::ComplexElement< KSRootRunModifier >( "ks_root_run_modifier" );

    STATICINT sKSRootRunModifierStructure =
            KSRootRunModifierBuilder::Attribute< std::string >( "name" ) +
            KSRootRunModifierBuilder::Attribute< std::string >( "add_modifier" );
}
