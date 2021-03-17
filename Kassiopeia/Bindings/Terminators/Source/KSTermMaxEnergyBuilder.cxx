#include "KSTermMaxEnergyBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTermMaxEnergyBuilder::~KComplexElement() = default;

STATICINT sKSTermMaxEnergyStructure =
    KSTermMaxEnergyBuilder::Attribute<std::string>("name") + KSTermMaxEnergyBuilder::Attribute<double>("energy");

STATICINT sKSTermMaxEnergy = KSRootBuilder::ComplexElement<KSTermMaxEnergy>("ksterm_max_energy");

}  // namespace katrin
