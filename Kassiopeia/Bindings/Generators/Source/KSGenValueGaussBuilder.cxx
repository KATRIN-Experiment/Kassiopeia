#include "KSGenValueGaussBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenValueGaussBuilder::~KComplexElement() {}

STATICINT sKSGenValueGaussStructure =
    KSGenValueGaussBuilder::Attribute<string>("name") + KSGenValueGaussBuilder::Attribute<double>("value_min") +
    KSGenValueGaussBuilder::Attribute<double>("value_max") + KSGenValueGaussBuilder::Attribute<double>("value_mean") +
    KSGenValueGaussBuilder::Attribute<double>("value_sigma");

STATICINT sKSGenValueGauss = KSRootBuilder::ComplexElement<KSGenValueGauss>("ksgen_value_gauss");

}  // namespace katrin
