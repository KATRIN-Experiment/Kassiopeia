#include "KSGenEnergyKryptonEventBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenEnergyKryptonEventBuilder::~KComplexElement() = default;

STATICINT sKSGenEnergyKryptonEventStructure =
    //! the name of this energy creator.
    KSGenEnergyKryptonEventBuilder::Attribute<std::string>("name") +
    //! if true, conversion electrons will always be generated.
    KSGenEnergyKryptonEventBuilder::Attribute<bool>("force_conversion") +
    //! if true, generation of conversion electrons is enabled.
    KSGenEnergyKryptonEventBuilder::Attribute<bool>("do_conversion") +
    //! if true, generation of an auger cascade is enabled.
    KSGenEnergyKryptonEventBuilder::Attribute<bool>("do_auger");


STATICINT sKSGenEnergyKryptonEvent =
    KSRootBuilder::ComplexElement<KSGenEnergyKryptonEvent>("ksgen_energy_krypton_event");

}  // namespace katrin
