#include "KSTermMaxEnergyBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTermMaxEnergyBuilder::~KComplexElement() {}

STATICINT sKSTermMaxEnergyStructure =
    KSTermMaxEnergyBuilder::Attribute<string>("name") + KSTermMaxEnergyBuilder::Attribute<double>("energy");

STATICINT sKSTermMaxEnergy = KSRootBuilder::ComplexElement<KSTermMaxEnergy>("ksterm_max_energy");

}  // namespace katrin
