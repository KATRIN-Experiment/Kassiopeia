#include "KSGenValueAngleSphericalBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenValueAngleSphericalBuilder::~KComplexElement() = default;

STATICINT sKSGenValueAngleSphericalStructure = KSGenValueAngleSphericalBuilder::Attribute<std::string>("name") +
                                               KSGenValueAngleSphericalBuilder::Attribute<double>("angle_min") +
                                               KSGenValueAngleSphericalBuilder::Attribute<double>("angle_max");

STATICINT sToolboxKSGenValueAngleSpherical =
    KSRootBuilder::ComplexElement<KSGenValueAngleSpherical>("ksgen_value_angle_spherical");

}  // namespace katrin
