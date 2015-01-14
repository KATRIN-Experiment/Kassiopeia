#include "KGShellPolyLineSurfaceBuilder.hh"
#include "KGInterfaceBuilder.hh"

namespace katrin
{

    static const int sKGShellPolyLineSurfaceBuilder =
        KGInterfaceBuilder::ComplexElement< KGShellPolyLineSurface >( "shell_poly_line_surface" );

    static const int sKGShellPolyLineSurfaceBuilderStructure =
        KGShellPolyLineSurfaceBuilder::Attribute< string >( "name" ) +
        KGShellPolyLineSurfaceBuilder::Attribute< double >( "angle_start" ) +
        KGShellPolyLineSurfaceBuilder::Attribute< double >( "angle_stop" ) +
        KGShellPolyLineSurfaceBuilder::Attribute< unsigned int >( "shell_mesh_count" ) +
        KGShellPolyLineSurfaceBuilder::Attribute< double >( "shell_mesh_power" ) +
        KGShellPolyLineSurfaceBuilder::ComplexElement< KGPlanarPolyLine >( "poly_line" );

}
