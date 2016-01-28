#include "KSTermTrappedBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTermTrappedBuilder::~KComplexElement()
    {
    }

    STATICINT sKSTermTrappedStructure =
            KSTermTrappedBuilder::Attribute< string >( "name" ) +
            KSTermTrappedBuilder::Attribute< unsigned int >( "max_turns" );

    STATICINT sKSTermTrapped =
            KSRootBuilder::ComplexElement< KSTermTrapped >( "ksterm_trapped" );

}
