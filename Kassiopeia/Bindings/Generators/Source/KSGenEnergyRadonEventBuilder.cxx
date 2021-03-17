#include "KSGenEnergyRadonEventBuilder.h"
//#include "KSGenGeneratorCompositeBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenEnergyRadonEventBuilder::~KComplexElement() = default;

STATICINT sKSGenEnergyRadonEventStructure = KSGenEnergyRadonEventBuilder::Attribute<std::string>("name") +
                                            KSGenEnergyRadonEventBuilder::Attribute<bool>("force_shake_off") +
                                            KSGenEnergyRadonEventBuilder::Attribute<bool>("force_conversion") +
                                            KSGenEnergyRadonEventBuilder::Attribute<bool>("do_shake_off") +
                                            KSGenEnergyRadonEventBuilder::Attribute<bool>("do_conversion") +
                                            KSGenEnergyRadonEventBuilder::Attribute<bool>("do_auger") +
                                            KSGenEnergyRadonEventBuilder::Attribute<int>("isotope_number");

STATICINT sKSGenEnergyRadonEvent = KSRootBuilder::ComplexElement<KSGenEnergyRadonEvent>("ksgen_energy_radon_event");

}  // namespace katrin
