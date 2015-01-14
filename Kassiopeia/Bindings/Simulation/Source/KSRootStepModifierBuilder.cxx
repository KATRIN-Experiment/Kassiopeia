#include "KSRootStepModifierBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{
    template< >
    KSRootStepModifierBuilder::~KComplexElement()
    {
    }

    static int sKSRootStepModifier =
            KSRootBuilder::ComplexElement< KSRootStepModifier >( "ks_root_stepmodifier" );

    static int sKSRootStepModifierStructure =
            KSRootStepModifierBuilder::Attribute< string >( "name" ) +
            KSRootStepModifierBuilder::Attribute< string >( "add_modifier" );
}