#include "KSGenValueSetBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenValueSetBuilder::~KComplexElement() = default;

STATICINT sKSGenValueSetStructure =
    //! the name of this value.
    KSGenValueSetBuilder::Attribute<std::string>("name") +
    //! the beginning of the interval.
    KSGenValueSetBuilder::Attribute<double>("value_start") +
    //! the end of the interval.
    KSGenValueSetBuilder::Attribute<double>("value_stop") + KSGenValueSetBuilder::Attribute<double>("value_increment") +
    //! the number of values to return.  must be at least two.
    KSGenValueSetBuilder::Attribute<unsigned int>("value_count");

STATICINT sKSGenValueSet = KSRootBuilder::ComplexElement<KSGenValueSet>("ksgen_value_set");

}  // namespace katrin
