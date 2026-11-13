#include "KSGenValueRadiusSphericalBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenValueRadiusSphericalBuilder::~KComplexElement() = default;

STATICINT sKSGenValueRadiusSphericalStructure =
    //! the name of this value.
    KSGenValueRadiusSphericalBuilder::Attribute<std::string>("name") +
    //! the minimum radius in meters to return. must be positive and less than radius_max.
    KSGenValueRadiusSphericalBuilder::Attribute<double>("radius_min") +
    //! the maximum radius in meters to return. must be positive and greater than radius_min.
    KSGenValueRadiusSphericalBuilder::Attribute<double>("radius_max");

STATICINT sToolboxKSGenValueRadiusSpherical =
    KSRootBuilder::ComplexElement<KSGenValueRadiusSpherical>("ksgen_value_radius_spherical");

}  // namespace katrin
