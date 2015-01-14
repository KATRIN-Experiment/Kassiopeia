#include "KGShellLineSegmentSurfaceBuilder.hh"
#include "KGInterfaceBuilder.hh"

namespace katrin
{

    static const int sKGShellLineSegmentSurfaceBuilder =
        KGInterfaceBuilder::ComplexElement< KGShellLineSegmentSurface >( "shell_line_segment_surface" );

    static const int sKGShellLineSegmentSurfaceBuilderStructure =
        KGShellLineSegmentSurfaceBuilder::Attribute< string >( "name" ) +
        KGShellLineSegmentSurfaceBuilder::Attribute< double >( "angle_start" ) +
        KGShellLineSegmentSurfaceBuilder::Attribute< double >( "angle_stop" ) +
        KGShellLineSegmentSurfaceBuilder::Attribute< unsigned int >( "shell_mesh_count" ) +
        KGShellLineSegmentSurfaceBuilder::Attribute< double >( "shell_mesh_power" ) +
        KGShellLineSegmentSurfaceBuilder::ComplexElement< KGPlanarLineSegment >( "line_segment" );

}
