#include "KSTermMinEnergyBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTermMinEnergyBuilder::~KComplexElement() {}

STATICINT sKSTermMinEnergyStructure =
    KSTermMinEnergyBuilder::Attribute<string>("name") + KSTermMinEnergyBuilder::Attribute<double>("energy");

STATICINT sKSTermMinEnergy = KSRootBuilder::ComplexElement<KSTermMinEnergy>("ksterm_min_energy");

}  // namespace katrin
