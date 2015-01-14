#include "KGTorusSpaceBuilder.hh"
#include "KGInterfaceBuilder.hh"

namespace katrin
{

    static const int sKGTorusSpaceBuilder =
        KGInterfaceBuilder::ComplexElement< KGTorusSpace >( "torus_space" );

    static const int sKGTorusSpaceBuilderStructure =
        KGTorusSpaceBuilder::Attribute< string >( "name" ) +
        KGTorusSpaceBuilder::Attribute< double >( "z" ) +
        KGTorusSpaceBuilder::Attribute< double >( "r" ) +
        KGTorusSpaceBuilder::Attribute< double >( "radius" ) +
        KGTorusSpaceBuilder::Attribute< unsigned int >( "toroidal_mesh_count" ) +
        KGTorusSpaceBuilder::Attribute< unsigned int >( "axial_mesh_count" );

}
