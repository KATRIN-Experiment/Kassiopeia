#include "KSRootStepModifierBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{
    template< >
    KSRootStepModifierBuilder::~KComplexElement()
    {
    }

    STATICINT sKSRootStepModifier =
            KSRootBuilder::ComplexElement< KSRootStepModifier >( "ks_root_step_modifier" );

    STATICINT sKSRootStepModifierStructure =
            KSRootStepModifierBuilder::Attribute< string >( "name" ) +
            KSRootStepModifierBuilder::Attribute< string >( "add_modifier" );
}
