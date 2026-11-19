#include "KSTermMaxLongEnergyBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTermMaxLongEnergyBuilder::~KComplexElement() = default;

STATICINT sKSTermMaxLongEnergyStructure =
    //! the name of the terminator.
    KSTermMaxLongEnergyBuilder::Attribute<std::string>("name") +
    //! the maximum allowed longitudinal kinetic energy.
    KSTermMaxLongEnergyBuilder::Attribute<double>("long_energy");

STATICINT sKSTermMaxLongEnergy = KSRootBuilder::ComplexElement<KSTermMaxLongEnergy>("ksterm_max_long_energy");

}  // namespace katrin
