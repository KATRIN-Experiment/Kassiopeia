#include "KSGenValueUniformBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenValueUniformBuilder::~KComplexElement() {}

STATICINT sKSGenValueUniformStructure = KSGenValueUniformBuilder::Attribute<string>("name") +
                                        KSGenValueUniformBuilder::Attribute<double>("value_min") +
                                        KSGenValueUniformBuilder::Attribute<double>("value_max");

STATICINT sToolboxKSGenValueUniform = KSRootBuilder::ComplexElement<KSGenValueUniform>("ksgen_value_uniform");

}  // namespace katrin
