#include "KSRootEventModifierBuilder.h"

#include "KSRootBuilder.h"

#include <string>

using namespace Kassiopeia;
namespace katrin
{
template<> KSRootEventModifierBuilder::~KComplexElement() {}

STATICINT sKSRootEventModifier = KSRootBuilder::ComplexElement<KSRootEventModifier>("ks_root_event_modifier");

STATICINT sKSRootEventModifierStructure = KSRootEventModifierBuilder::Attribute<std::string>("name") +
                                          KSRootEventModifierBuilder::Attribute<std::string>("add_modifier");
}  // namespace katrin
