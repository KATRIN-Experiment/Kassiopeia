#include "KSGenValueRadiusFractionBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenValueRadiusFractionBuilder::~KComplexElement() = default;

STATICINT sKSGenValueRadiusFractionStructure = KSGenValueRadiusFractionBuilder::Attribute<std::string>("name");

STATICINT sToolboxKSGenValueRadiusFraction =
    KSRootBuilder::ComplexElement<KSGenValueRadiusFraction>("ksgen_value_radius_fraction");

}  // namespace katrin
