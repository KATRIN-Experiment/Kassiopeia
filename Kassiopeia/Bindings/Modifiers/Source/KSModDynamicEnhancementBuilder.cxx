#include "KSModDynamicEnhancementBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{
template<> KSModDynamicEnhancementBuilder::~KComplexElement() = default;

STATICINT SKSModDynamicEnhancementStructure = KSModDynamicEnhancementBuilder::Attribute<std::string>("name") +
                                              KSModDynamicEnhancementBuilder::Attribute<std::string>("synchrotron") +
                                              KSModDynamicEnhancementBuilder::Attribute<std::string>("scattering") +
                                              KSModDynamicEnhancementBuilder::Attribute<double>("static_enhancement") +
                                              KSModDynamicEnhancementBuilder::Attribute<bool>("dynamic") +
                                              KSModDynamicEnhancementBuilder::Attribute<double>("reference_energy");

STATICINT sKSModDynamicEnhancement =
    KSRootBuilder::ComplexElement<KSModDynamicEnhancement>("ksmod_dynamic_enhancement");
}  // namespace katrin
