#include "KSGenValueAngleSphericalBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenValueAngleSphericalBuilder::~KComplexElement() = default;

STATICINT sKSGenValueAngleSphericalStructure =
    //! the name of this value.
    KSGenValueAngleSphericalBuilder::Attribute<std::string>("name") +
    //! the minimum angle in degrees to return. must be greater than zero and less than angle_max and less than 90.
    KSGenValueAngleSphericalBuilder::Attribute<double>("angle_min") +
    //! the maximum angle in degrees to return. must be greater than zero and greater than angle_max and less than 90.
    KSGenValueAngleSphericalBuilder::Attribute<double>("angle_max");

STATICINT sToolboxKSGenValueAngleSpherical =
    KSRootBuilder::ComplexElement<KSGenValueAngleSpherical>("ksgen_value_angle_spherical");

}  // namespace katrin
