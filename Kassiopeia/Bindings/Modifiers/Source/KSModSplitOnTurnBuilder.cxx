#include "KSModSplitOnTurnBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{
    template< >
    KSModSplitOnTurnBuilder::~KComplexElement()
    {
    }

    STATICINT SKSModSplitOnTurnStructure =
            KSModSplitOnTurnBuilder::Attribute< string >( "name" ) +
            KSModSplitOnTurnBuilder::Attribute< string >( "direction" );

    STATICINT sKSModSplitOnTurn =
            KSRootBuilder::ComplexElement< KSModSplitOnTurn >( "ksmod_split_on_turn" );
}
