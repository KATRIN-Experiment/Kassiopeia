#include "KSTermMinLongEnergyBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTermMinLongEnergyBuilder::~KComplexElement() {}

STATICINT sKSTermMinLongEnergyStructure = KSTermMinLongEnergyBuilder::Attribute<string>("name") +
                                          KSTermMinLongEnergyBuilder::Attribute<double>("long_energy");

STATICINT sKSTermMinLongEnergy = KSRootBuilder::ComplexElement<KSTermMinLongEnergy>("ksterm_min_long_energy");

}  // namespace katrin
