#include "KSTermTrappedBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSTermTrappedBuilder::~KComplexElement()
    {
    }

    STATICINT sKSTermTrappedStructure =
            KSTermTrappedBuilder::Attribute< string >( "name" ) +
            KSTermTrappedBuilder::Attribute< int >( "max_turns" );

    STATICINT sKSTermTrapped =
            KSRootBuilder::ComplexElement< KSTermTrapped >( "ksterm_trapped" );

}
