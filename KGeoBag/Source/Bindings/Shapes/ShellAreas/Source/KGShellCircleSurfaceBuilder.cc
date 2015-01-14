#include "KGShellCircleSurfaceBuilder.hh"
#include "KGInterfaceBuilder.hh"

namespace katrin
{

    static const int sKGShellCircleSurfaceBuilder =
        KGInterfaceBuilder::ComplexElement< KGShellCircleSurface >( "shell_circle_surface" );

    static const int sKGShellCircleSurfaceBuilderStructure =
        KGShellCircleSurfaceBuilder::Attribute< string >( "name" ) +
        KGShellCircleSurfaceBuilder::Attribute< double >( "angle_start" ) +
        KGShellCircleSurfaceBuilder::Attribute< double >( "angle_stop" ) +
        KGShellCircleSurfaceBuilder::Attribute< unsigned int >( "shell_mesh_count" ) +
        KGShellCircleSurfaceBuilder::Attribute< double >( "shell_mesh_power" ) +
        KGShellCircleSurfaceBuilder::ComplexElement< KGPlanarCircle >( "circle" );

}
