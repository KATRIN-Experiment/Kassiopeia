#include "KSRootStepModifierBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{
template<> KSRootStepModifierBuilder::~KComplexElement() = default;

STATICINT sKSRootStepModifier = KSRootBuilder::ComplexElement<KSRootStepModifier>("ks_root_step_modifier");

STATICINT sKSRootStepModifierStructure = KSRootStepModifierBuilder::Attribute<std::string>("name") +
                                         KSRootStepModifierBuilder::Attribute<std::string>("add_modifier");
}  // namespace katrin
