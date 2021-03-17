#include "KSGenValueRadiusSphericalBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenValueRadiusSphericalBuilder::~KComplexElement() = default;

STATICINT sKSGenValueRadiusSphericalStructure = KSGenValueRadiusSphericalBuilder::Attribute<std::string>("name") +
                                                KSGenValueRadiusSphericalBuilder::Attribute<double>("radius_min") +
                                                KSGenValueRadiusSphericalBuilder::Attribute<double>("radius_max");

STATICINT sToolboxKSGenValueRadiusSpherical =
    KSRootBuilder::ComplexElement<KSGenValueRadiusSpherical>("ksgen_value_radius_spherical");

}  // namespace katrin
