#include "KSIntDensityConstantBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSIntDensityConstantBuilder::~KComplexElement() = default;

STATICINT sKSIntDensityConstantStructure =
    //! the name of the density.
    KSIntDensityConstantBuilder::Attribute<std::string>("name") +
    //! the ambient temperature in kelvin.
    KSIntDensityConstantBuilder::Attribute<double>("temperature") +
    //! the ambient pressure in millibar.
    KSIntDensityConstantBuilder::Attribute<double>("pressure") +
    KSIntDensityConstantBuilder::Attribute<double>("pressure_mbar") +
    KSIntDensityConstantBuilder::Attribute<double>("density");

STATICINT sToolboxKSIntDensityConstant = KSRootBuilder::ComplexElement<KSIntDensityConstant>("ksint_density_constant");
}  // namespace katrin
