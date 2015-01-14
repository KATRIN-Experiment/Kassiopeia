#include "KGBoxSpaceBuilder.hh"
#include "KGInterfaceBuilder.hh"

namespace katrin
{

    static const int sKGBoxSpaceBuilder =
        KGInterfaceBuilder::ComplexElement< KGBoxSpace >( "box_space" );

    static const int sKGBoxSpaceBuilderStructure =
        KGBoxSpaceBuilder::Attribute< string >( "name" ) +
        KGBoxSpaceBuilder::Attribute< double >( "xa" ) +
        KGBoxSpaceBuilder::Attribute< double >( "xb" ) +
        KGBoxSpaceBuilder::Attribute< unsigned int >( "x_mesh_count" ) +
        KGBoxSpaceBuilder::Attribute< double >( "x_mesh_power" ) +
        KGBoxSpaceBuilder::Attribute< double >( "ya" ) +
        KGBoxSpaceBuilder::Attribute< double >( "yb" ) +
        KGBoxSpaceBuilder::Attribute< unsigned int >( "y_mesh_count" ) +
        KGBoxSpaceBuilder::Attribute< double >( "y_mesh_power" ) +
        KGBoxSpaceBuilder::Attribute< double >( "za" ) +
        KGBoxSpaceBuilder::Attribute< double >( "zb" ) +
        KGBoxSpaceBuilder::Attribute< unsigned int >( "z_mesh_count" ) +
        KGBoxSpaceBuilder::Attribute< double >( "z_mesh_power" );

}
