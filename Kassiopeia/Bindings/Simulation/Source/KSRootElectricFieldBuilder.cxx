#include "KSRootElectricFieldBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSRootElectricFieldBuilder::~KComplexElement() = default;

STATICINT sKSRootElectricField = KSRootBuilder::ComplexElement<KSRootElectricField>("ks_root_electric_field");

STATICINT sKSRootElectricFieldStructure = KSRootElectricFieldBuilder::Attribute<std::string>("name") +
                                          KSRootElectricFieldBuilder::Attribute<std::string>("add_electric_field");

}  // namespace katrin
