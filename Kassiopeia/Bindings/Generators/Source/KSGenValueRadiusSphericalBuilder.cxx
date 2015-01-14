#include "KSGenValueRadiusSphericalBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSGenValueRadiusSphericalBuilder::~KComplexElement()
    {
    }

    static int sKSGenValueRadiusSphericalStructure =
        KSGenValueRadiusSphericalBuilder::Attribute< string >( "name" ) +
        KSGenValueRadiusSphericalBuilder::Attribute< double >( "radius_min" ) +
        KSGenValueRadiusSphericalBuilder::Attribute< double >( "radius_max" );

    static int sToolboxKSGenValueRadiusSpherical =
        KSRootBuilder::ComplexElement< KSGenValueRadiusSpherical >( "ksgen_value_radius_spherical" );

}
