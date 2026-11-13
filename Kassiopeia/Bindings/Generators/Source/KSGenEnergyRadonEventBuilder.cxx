#include "KSGenEnergyRadonEventBuilder.h"
//#include "KSGenGeneratorCompositeBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenEnergyRadonEventBuilder::~KComplexElement() = default;

STATICINT sKSGenEnergyRadonEventStructure =
    //! the name of this energy creator.
    KSGenEnergyRadonEventBuilder::Attribute<std::string>("name") +
    //! if true, shake off electrons will always be generated.
    KSGenEnergyRadonEventBuilder::Attribute<bool>("force_shake_off") +
    //! if true, conversion electrons will always be generated.
    KSGenEnergyRadonEventBuilder::Attribute<bool>("force_conversion") +
    //! if true, generation of shake off electrons is enabled.
    KSGenEnergyRadonEventBuilder::Attribute<bool>("do_shake_off") +
    //! if true, generation of conversion electrons is enabled.
    KSGenEnergyRadonEventBuilder::Attribute<bool>("do_conversion") +
    //! if true, generation of an auger cascade is enabled.
    KSGenEnergyRadonEventBuilder::Attribute<bool>("do_auger") +
    //! may be 219 or 220.
    KSGenEnergyRadonEventBuilder::Attribute<int>("isotope_number");

STATICINT sKSGenEnergyRadonEvent = KSRootBuilder::ComplexElement<KSGenEnergyRadonEvent>("ksgen_energy_radon_event");

}  // namespace katrin
