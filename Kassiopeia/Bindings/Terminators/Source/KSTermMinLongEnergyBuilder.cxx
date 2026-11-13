#include "KSTermMinLongEnergyBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTermMinLongEnergyBuilder::~KComplexElement() = default;

STATICINT sKSTermMinLongEnergyStructure =
    //! the name of the terminator.
    KSTermMinLongEnergyBuilder::Attribute<std::string>("name") +
    //! the minimum allowed longitudinal kinetic energy.
    KSTermMinLongEnergyBuilder::Attribute<double>("long_energy");

STATICINT sKSTermMinLongEnergy = KSRootBuilder::ComplexElement<KSTermMinLongEnergy>("ksterm_min_long_energy");

}  // namespace katrin
