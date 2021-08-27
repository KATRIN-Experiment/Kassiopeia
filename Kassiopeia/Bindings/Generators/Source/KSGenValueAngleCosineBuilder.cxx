#include "KSGenValueAngleCosineBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenValueAngleCosineBuilder::~KComplexElement() = default;

STATICINT sKSGenValueAngleCosineStructure = KSGenValueAngleCosineBuilder::Attribute<std::string>("name") +
                                            KSGenValueAngleCosineBuilder::Attribute<std::string>("mode") +
                                            KSGenValueAngleCosineBuilder::Attribute<double>("angle_min") +
                                            KSGenValueAngleCosineBuilder::Attribute<double>("angle_max");

STATICINT sToolboxKSGenValueAngleCosine =
    KSRootBuilder::ComplexElement<KSGenValueAngleCosine>("ksgen_value_angle_cosine");

}  // namespace katrin
