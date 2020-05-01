#include "KSGenValueListBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenValueListBuilder::~KComplexElement() {}

STATICINT sKSGenValueListStructure =
    KSGenValueListBuilder::Attribute<string>("name") + KSGenValueListBuilder::Attribute<double>("add_value");

STATICINT sKSGenValueList = KSRootBuilder::ComplexElement<KSGenValueList>("ksgen_value_list");

}  // namespace katrin
