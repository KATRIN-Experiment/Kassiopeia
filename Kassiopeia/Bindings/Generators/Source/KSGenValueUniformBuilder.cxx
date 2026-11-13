#include "KSGenValueUniformBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenValueUniformBuilder::~KComplexElement() = default;

STATICINT sKSGenValueUniformStructure =
    //! the name of this value.
    KSGenValueUniformBuilder::Attribute<std::string>("name") +
    //! the minimum value to return.
    KSGenValueUniformBuilder::Attribute<double>("value_min") +
    //! the maximum value to return.
    KSGenValueUniformBuilder::Attribute<double>("value_max");

STATICINT sToolboxKSGenValueUniform = KSRootBuilder::ComplexElement<KSGenValueUniform>("ksgen_value_uniform");

}  // namespace katrin
