#include "KSGenValueRadiusCylindricalBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenValueRadiusCylindricalBuilder::~KComplexElement() = default;

STATICINT sKSGenValueRadiusCylindricalStructure = KSGenValueRadiusCylindricalBuilder::Attribute<std::string>("name") +
                                                  KSGenValueRadiusCylindricalBuilder::Attribute<double>("radius_min") +
                                                  KSGenValueRadiusCylindricalBuilder::Attribute<double>("radius_max");

STATICINT sToolboxKSGenValueRadiusCylindrical =
    KSRootBuilder::ComplexElement<KSGenValueRadiusCylindrical>("ksgen_value_radius_cylindrical");

}  // namespace katrin
