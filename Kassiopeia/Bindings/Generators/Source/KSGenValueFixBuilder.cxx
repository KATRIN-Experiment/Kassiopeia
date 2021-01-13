#include "KSGenValueFixBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenValueFixBuilder::~KComplexElement() = default;

STATICINT sKSGenValueFixStructure =
    KSGenValueFixBuilder::Attribute<std::string>("name") + KSGenValueFixBuilder::Attribute<double>("value");

STATICINT sKSGenValueFix = KSRootBuilder::ComplexElement<KSGenValueFix>("ksgen_value_fix");

}  // namespace katrin
