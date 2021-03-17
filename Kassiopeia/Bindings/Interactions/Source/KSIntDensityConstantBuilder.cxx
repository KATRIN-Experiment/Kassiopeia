#include "KSIntDensityConstantBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSIntDensityConstantBuilder::~KComplexElement() = default;

STATICINT sKSIntDensityConstantStructure = KSIntDensityConstantBuilder::Attribute<std::string>("name") +
                                           KSIntDensityConstantBuilder::Attribute<double>("temperature") +
                                           KSIntDensityConstantBuilder::Attribute<double>("pressure") +
                                           KSIntDensityConstantBuilder::Attribute<double>("pressure_mbar") +
                                           KSIntDensityConstantBuilder::Attribute<double>("density");

STATICINT sToolboxKSIntDensityConstant = KSRootBuilder::ComplexElement<KSIntDensityConstant>("ksint_density_constant");
}  // namespace katrin
