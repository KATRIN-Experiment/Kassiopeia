#include "KGTorusSurfaceBuilder.hh"
#include "KGInterfaceBuilder.hh"

namespace katrin
{

    static const int sKGTorusSurfaceBuilder =
        KGInterfaceBuilder::ComplexElement< KGTorusSurface >( "torus_surface" );

    static const int sKGTorusSurfaceBuilderStructure =
        KGTorusSurfaceBuilder::Attribute< string >( "name" ) +
        KGTorusSurfaceBuilder::Attribute< double >( "z" ) +
        KGTorusSurfaceBuilder::Attribute< double >( "r" ) +
        KGTorusSurfaceBuilder::Attribute< double >( "radius" ) +
        KGTorusSurfaceBuilder::Attribute< unsigned int >( "toroidal_mesh_count" ) +
        KGTorusSurfaceBuilder::Attribute< unsigned int >( "axial_mesh_count" );

}
