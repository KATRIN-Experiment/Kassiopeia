#include "KSGenValueRadiusCylindricalBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenValueRadiusCylindricalBuilder::~KComplexElement() {}

STATICINT sKSGenValueRadiusCylindricalStructure = KSGenValueRadiusCylindricalBuilder::Attribute<string>("name") +
                                                  KSGenValueRadiusCylindricalBuilder::Attribute<double>("radius_min") +
                                                  KSGenValueRadiusCylindricalBuilder::Attribute<double>("radius_max");

STATICINT sToolboxKSGenValueRadiusCylindrical =
    KSRootBuilder::ComplexElement<KSGenValueRadiusCylindrical>("ksgen_value_radius_cylindrical");

}  // namespace katrin
