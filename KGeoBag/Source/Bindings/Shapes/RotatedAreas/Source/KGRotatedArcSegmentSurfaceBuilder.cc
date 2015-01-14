#include "KGRotatedArcSegmentSurfaceBuilder.hh"
#include "KGInterfaceBuilder.hh"

namespace katrin
{

    static const int sKGRotatedArcSegmentSurfaceBuilder =
        KGInterfaceBuilder::ComplexElement< KGRotatedArcSegmentSurface >( "rotated_arc_segment_surface" );

    static const int sKGRotatedArcSegmentSurfaceBuilderStructure =
        KGRotatedArcSegmentSurfaceBuilder::Attribute< string >( "name" ) +
        KGRotatedArcSegmentSurfaceBuilder::Attribute< unsigned int >( "rotated_mesh_count" ) +
        KGRotatedArcSegmentSurfaceBuilder::ComplexElement< KGPlanarArcSegment >( "arc_segment" );

}
