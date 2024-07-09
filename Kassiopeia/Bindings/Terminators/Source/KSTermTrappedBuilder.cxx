#include "KSTermTrappedBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTermTrappedBuilder::~KComplexElement() = default;

STATICINT sKSTermTrappedStructure =
    KSTermTrappedBuilder::Attribute<std::string>("name") +
    KSTermTrappedBuilder::Attribute<bool>("use_electric_field") +
    KSTermTrappedBuilder::Attribute<bool>("use_magnetic_field") +
    KSTermTrappedBuilder::Attribute<int>("max_turns");

STATICINT sKSTermTrapped = KSRootBuilder::ComplexElement<KSTermTrapped>("ksterm_trapped");

}  // namespace katrin
