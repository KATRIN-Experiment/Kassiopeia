#include "KSTermTrappedBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTermTrappedBuilder::~KComplexElement()
    {
    }

    static int sKSTermTrappedStructure =
            KSTermTrappedBuilder::Attribute< string >( "name" ) +
            KSTermTrappedBuilder::Attribute< unsigned int >( "max_turns" );

    static int sKSTermTrapped =
            KSRootBuilder::ComplexElement< KSTermTrapped >( "ksterm_trapped" );

}
