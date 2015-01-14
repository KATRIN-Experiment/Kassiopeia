#include "KSGeoSurfaceBuilder.h"
#include "KSCommandMemberBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    static int sKSGeoSurfaceStructure =
        KSGeoSurfaceBuilder::Attribute< string >( "name" ) +
        KSGeoSurfaceBuilder::Attribute< string >( "surfaces" ) +
        KSGeoSurfaceBuilder::Attribute< string >( "spaces" ) +
        KSGeoSurfaceBuilder::Attribute< string >( "command" ) +
        KSGeoSurfaceBuilder::ComplexElement< KSCommandMemberData >( "command" );

    static int sKSGeoSurface =
        KSRootBuilder::ComplexElement< KSGeoSurface >( "ksgeo_surface" );

}
