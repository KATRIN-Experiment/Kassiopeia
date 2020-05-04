#include "KSGenValueSetBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenValueSetBuilder::~KComplexElement() {}

STATICINT sKSGenValueSetStructure =
    KSGenValueSetBuilder::Attribute<string>("name") + KSGenValueSetBuilder::Attribute<double>("value_start") +
    KSGenValueSetBuilder::Attribute<double>("value_stop") + KSGenValueSetBuilder::Attribute<double>("value_increment") +
    KSGenValueSetBuilder::Attribute<unsigned int>("value_count");

STATICINT sKSGenValueSet = KSRootBuilder::ComplexElement<KSGenValueSet>("ksgen_value_set");

}  // namespace katrin
