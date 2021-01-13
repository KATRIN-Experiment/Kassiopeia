#include "KSGenValueListBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenValueListBuilder::~KComplexElement() = default;

STATICINT sKSGenValueListStructure = KSGenValueListBuilder::Attribute<std::string>("name") +
                                     KSGenValueListBuilder::Attribute<double>("add_value") +
                                     KSGenValueListBuilder::Attribute<bool>("randomize");

STATICINT sKSGenValueList = KSRootBuilder::ComplexElement<KSGenValueList>("ksgen_value_list");

}  // namespace katrin
