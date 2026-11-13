#include "KSTermMinEnergyBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTermMinEnergyBuilder::~KComplexElement() = default;

STATICINT sKSTermMinEnergyStructure =
    //! the name of the terminator.
    KSTermMinEnergyBuilder::Attribute<std::string>("name") +
    //! the minimum allowed kinetic energy.
    KSTermMinEnergyBuilder::Attribute<double>("energy");

STATICINT sKSTermMinEnergy = KSRootBuilder::ComplexElement<KSTermMinEnergy>("ksterm_min_energy");

}  // namespace katrin
