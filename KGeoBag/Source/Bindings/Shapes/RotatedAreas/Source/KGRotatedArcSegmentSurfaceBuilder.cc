#include "KGRotatedArcSegmentSurfaceBuilder.hh"
#include "KGInterfaceBuilder.hh"

namespace katrin
{

    STATICINT sKGRotatedArcSegmentSurfaceBuilder =
        KGInterfaceBuilder::ComplexElement< KGRotatedArcSegmentSurface >( "rotated_arc_segment_surface" );

    STATICINT sKGRotatedArcSegmentSurfaceBuilderStructure =
        KGRotatedArcSegmentSurfaceBuilder::Attribute< string >( "name" ) +
        KGRotatedArcSegmentSurfaceBuilder::Attribute< unsigned int >( "rotated_mesh_count" ) +
        KGRotatedArcSegmentSurfaceBuilder::ComplexElement< KGPlanarArcSegment >( "arc_segment" );

}
