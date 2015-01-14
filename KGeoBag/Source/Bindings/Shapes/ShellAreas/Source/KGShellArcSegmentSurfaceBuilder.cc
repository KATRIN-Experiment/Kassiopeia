#include "KGShellArcSegmentSurfaceBuilder.hh"
#include "KGInterfaceBuilder.hh"

namespace katrin
{

    static const int sKGShellArcSegmentSurfaceBuilder =
        KGInterfaceBuilder::ComplexElement< KGShellArcSegmentSurface >( "shell_arc_segment_surface" );

    static const int sKGShellArcSegmentSurfaceBuilderStructure =
        KGShellArcSegmentSurfaceBuilder::Attribute< string >( "name" ) +
        KGShellArcSegmentSurfaceBuilder::Attribute< double >( "angle_start" ) +
        KGShellArcSegmentSurfaceBuilder::Attribute< double >( "angle_stop" ) +
        KGShellArcSegmentSurfaceBuilder::Attribute< unsigned int >( "shell_mesh_count" ) +
        KGShellArcSegmentSurfaceBuilder::Attribute< double >( "shell_mesh_power" ) +
        KGShellArcSegmentSurfaceBuilder::ComplexElement< KGPlanarArcSegment >( "arc_segment" );

}
