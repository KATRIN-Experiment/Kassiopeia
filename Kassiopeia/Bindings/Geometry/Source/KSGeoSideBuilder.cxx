#include "KSGeoSideBuilder.h"
#include "KSCommandMemberBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    static int sKSGeoSideStructure =
        KSGeoSideBuilder::Attribute< string >( "name" ) +
        KSGeoSideBuilder::Attribute< string >( "surfaces" ) +
        KSGeoSideBuilder::Attribute< string >( "spaces" ) +
        KSGeoSideBuilder::Attribute< string >( "command" ) +
        KSGeoSideBuilder::ComplexElement< KSCommandMemberData >( "command" );

    static int sKSGeoSide =
        KSRootBuilder::ComplexElement< KSGeoSide >( "ksgeo_side" );

}
