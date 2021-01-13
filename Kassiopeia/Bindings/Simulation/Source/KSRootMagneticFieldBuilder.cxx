#include "KSRootMagneticFieldBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSRootMagneticFieldBuilder::~KComplexElement() = default;

STATICINT sKSRootMagneticField = KSRootBuilder::ComplexElement<KSRootMagneticField>("ks_root_magnetic_field");

STATICINT sKSRootMagneticFieldStructure = KSRootMagneticFieldBuilder::Attribute<std::string>("name") +
                                          KSRootMagneticFieldBuilder::Attribute<std::string>("add_magnetic_field");

}  // namespace katrin
