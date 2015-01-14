#include "KGCutConeTubeSpaceBuilder.hh"
#include "KGInterfaceBuilder.hh"

namespace katrin
{

    static const int sKGCutConeTubeSpaceBuilder =
        KGInterfaceBuilder::ComplexElement< KGCutConeTubeSpace >( "cut_cone_tube_space" );

    static const int sKGCutConeTubeSpaceBuilderStructure =
        KGCutConeTubeSpaceBuilder::Attribute< string >( "name" ) +
        KGCutConeTubeSpaceBuilder::Attribute< double >( "z1" ) +
        KGCutConeTubeSpaceBuilder::Attribute< double >( "z2" ) +
        KGCutConeTubeSpaceBuilder::Attribute< double >( "r11" ) +
        KGCutConeTubeSpaceBuilder::Attribute< double >( "r12" ) +
        KGCutConeTubeSpaceBuilder::Attribute< double >( "r21" ) +
        KGCutConeTubeSpaceBuilder::Attribute< double >( "r22" ) +
        KGCutConeTubeSpaceBuilder::Attribute< unsigned int >( "radial_mesh_count" ) +
        KGCutConeTubeSpaceBuilder::Attribute< double >( "radial_mesh_power" ) +
        KGCutConeTubeSpaceBuilder::Attribute< unsigned int >( "longitudinal_mesh_count" ) +
        KGCutConeTubeSpaceBuilder::Attribute< double >( "longitudinal_mesh_power" ) +
        KGCutConeTubeSpaceBuilder::Attribute< unsigned int >( "axial_mesh_count" );

}
