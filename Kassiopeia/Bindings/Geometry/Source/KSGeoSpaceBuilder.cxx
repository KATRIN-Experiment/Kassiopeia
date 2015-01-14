#include "KSGeoSpaceBuilder.h"
#include "KSGeoSurfaceBuilder.h"
#include "KSGeoSideBuilder.h"
#include "KSCommandMemberBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    static int sKSSpaceStructure =
        KSGeoSpaceBuilder::Attribute< string >( "name" ) +
        KSGeoSpaceBuilder::Attribute< string >( "spaces" ) +
        KSGeoSpaceBuilder::Attribute< string >( "command" ) +
        KSGeoSpaceBuilder::ComplexElement< KSCommandMemberData >( "command" ) +
        KSGeoSpaceBuilder::ComplexElement< KSGeoSpace >( "geo_space" ) +
        KSGeoSpaceBuilder::ComplexElement< KSGeoSurface >( "geo_surface" ) +
        KSGeoSpaceBuilder::ComplexElement< KSGeoSide >( "geo_side" );

    static int sKSSpace =
        KSRootBuilder::ComplexElement< KSGeoSpace >( "ksgeo_space" );

}
