#include "KSGenValueAngleSphericalBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSGenValueAngleSphericalBuilder::~KComplexElement()
    {
    }

    STATICINT sKSGenValueAngleSphericalStructure =
        KSGenValueAngleSphericalBuilder::Attribute< string >( "name" ) +
        KSGenValueAngleSphericalBuilder::Attribute< double >( "angle_min" ) +
        KSGenValueAngleSphericalBuilder::Attribute< double >( "angle_max" );

    STATICINT sToolboxKSGenValueAngleSpherical =
        KSRootBuilder::ComplexElement< KSGenValueAngleSpherical >( "ksgen_value_angle_spherical" );

}
