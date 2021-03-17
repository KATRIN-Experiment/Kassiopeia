#include "KSGenValueParetoBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenValueParetoBuilder::~KComplexElement() = default;

STATICINT sKSGenValueParetoStructure =
    KSGenValueParetoBuilder::Attribute<std::string>("name") + KSGenValueParetoBuilder::Attribute<double>("value_min") +
    KSGenValueParetoBuilder::Attribute<double>("value_max") + KSGenValueParetoBuilder::Attribute<double>("slope") +
    KSGenValueParetoBuilder::Attribute<double>("cutoff") + KSGenValueParetoBuilder::Attribute<double>("offset");

STATICINT sKSGenValuePareto = KSRootBuilder::ComplexElement<KSGenValuePareto>("ksgen_value_pareto");

}  // namespace katrin
