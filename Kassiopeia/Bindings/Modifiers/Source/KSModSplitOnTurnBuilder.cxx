#include "KSModSplitOnTurnBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{
template<> KSModSplitOnTurnBuilder::~KComplexElement() = default;

STATICINT SKSModSplitOnTurnStructure = KSModSplitOnTurnBuilder::Attribute<std::string>("name") +
                                       KSModSplitOnTurnBuilder::Attribute<std::string>("direction");

STATICINT sKSModSplitOnTurn = KSRootBuilder::ComplexElement<KSModSplitOnTurn>("ksmod_split_on_turn");
}  // namespace katrin
