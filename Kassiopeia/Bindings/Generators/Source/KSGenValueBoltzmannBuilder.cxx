#include "KSGenValueBoltzmannBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenValueBoltzmannBuilder::~KComplexElement() {}

STATICINT sKSGenValueBoltzmannStructure = KSGenValueBoltzmannBuilder::Attribute<string>("name") +
                                          KSGenValueBoltzmannBuilder::Attribute<double>("value_mass") +
                                          KSGenValueBoltzmannBuilder::Attribute<double>("value_kT") +
                                          KSGenValueBoltzmannBuilder::Attribute<bool>("unit_eV");

STATICINT sKSGenValueBoltzmann = KSRootBuilder::ComplexElement<KSGenValueBoltzmann>("ksgen_value_boltzmann");

}  // namespace katrin
