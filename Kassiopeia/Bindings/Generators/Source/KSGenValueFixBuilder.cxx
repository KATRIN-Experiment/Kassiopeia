#include "KSGenValueFixBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenValueFixBuilder::~KComplexElement() = default;

STATICINT sKSGenValueFixStructure =
    //! the name of this value.
    KSGenValueFixBuilder::Attribute<std::string>("name") +
    //! the value to be returned.
    KSGenValueFixBuilder::Attribute<double>("value");

STATICINT sKSGenValueFix = KSRootBuilder::ComplexElement<KSGenValueFix>("ksgen_value_fix");

}  // namespace katrin
