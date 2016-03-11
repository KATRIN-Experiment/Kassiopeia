#include "KGConeSurfaceBuilder.hh"
#include "KGInterfaceBuilder.hh"

namespace katrin
{

    STATICINT sKGConeSurfaceBuilder =
        KGInterfaceBuilder::ComplexElement< KGConeSurface >( "cone_surface" );

    STATICINT sKGConeSurfaceBuilderStructure =
        KGConeSurfaceBuilder::Attribute< string >( "name" ) +
        KGConeSurfaceBuilder::Attribute< double >( "za" ) +
        KGConeSurfaceBuilder::Attribute< double >( "zb" ) +
        KGConeSurfaceBuilder::Attribute< double >( "rb" ) +
        KGConeSurfaceBuilder::Attribute< unsigned int >( "longitudinal_mesh_count" ) +
        KGConeSurfaceBuilder::Attribute< double >( "longitudinal_mesh_power" ) +
        KGConeSurfaceBuilder::Attribute< unsigned int >( "axial_mesh_count" );

}
