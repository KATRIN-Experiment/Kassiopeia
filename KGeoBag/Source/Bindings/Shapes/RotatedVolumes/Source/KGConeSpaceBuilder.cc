#include "KGConeSpaceBuilder.hh"
#include "KGInterfaceBuilder.hh"

namespace katrin
{

    STATICINT sKGConeSpaceBuilder =
        KGInterfaceBuilder::ComplexElement< KGConeSpace >( "cone_space" );

    STATICINT sKGConeSpaceBuilderStructure =
        KGConeSpaceBuilder::Attribute< string >( "name" ) +
        KGConeSpaceBuilder::Attribute< double >( "za" ) +
        KGConeSpaceBuilder::Attribute< double >( "zb" ) +
        KGConeSpaceBuilder::Attribute< double >( "rb" ) +
        KGConeSpaceBuilder::Attribute< unsigned int >( "longitudinal_mesh_count" ) +
        KGConeSpaceBuilder::Attribute< double >( "longitudinal_mesh_power" ) +
        KGConeSpaceBuilder::Attribute< unsigned int >( "radial_mesh_count" ) +
        KGConeSpaceBuilder::Attribute< double >( "radial_mesh_power" ) +
        KGConeSpaceBuilder::Attribute< unsigned int >( "axial_mesh_count" );

}
