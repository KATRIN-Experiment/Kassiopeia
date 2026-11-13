#include "KSGenValueRadiusCylindricalBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenValueRadiusCylindricalBuilder::~KComplexElement() = default;

STATICINT sKSGenValueRadiusCylindricalStructure =
    //! the name of this value.
    KSGenValueRadiusCylindricalBuilder::Attribute<std::string>("name") +
    //! the minimum radius in meters to return. must be positive and less than radius_max.
    KSGenValueRadiusCylindricalBuilder::Attribute<double>("radius_min") +
    //! the maximum radius in meters to return. must be positive and greater than radius_min.
    KSGenValueRadiusCylindricalBuilder::Attribute<double>("radius_max");

STATICINT sToolboxKSGenValueRadiusCylindrical =
    KSRootBuilder::ComplexElement<KSGenValueRadiusCylindrical>("ksgen_value_radius_cylindrical");

}  // namespace katrin
