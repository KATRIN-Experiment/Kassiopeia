#include "KSGenEnergyKryptonEventBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenEnergyKryptonEventBuilder::~KComplexElement() = default;

STATICINT sKSGenEnergyKryptonEventStructure = KSGenEnergyKryptonEventBuilder::Attribute<std::string>("name") +
                                              KSGenEnergyKryptonEventBuilder::Attribute<bool>("force_conversion") +
                                              KSGenEnergyKryptonEventBuilder::Attribute<bool>("do_conversion") +
                                              KSGenEnergyKryptonEventBuilder::Attribute<bool>("do_auger");


STATICINT sKSGenEnergyKryptonEvent =
    KSRootBuilder::ComplexElement<KSGenEnergyKryptonEvent>("ksgen_energy_krypton_event");

}  // namespace katrin
