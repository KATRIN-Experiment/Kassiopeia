#include "KGRotatedLineSegmentSpaceBuilder.hh"
#include "KGInterfaceBuilder.hh"

namespace katrin
{

    static const int sKGRotatedLineSegmentSpaceBuilder =
        KGInterfaceBuilder::ComplexElement< KGRotatedLineSegmentSpace >( "rotated_line_segment_space" );

    static const int sKGRotatedLineSegmentSpaceBuilderStructure =
        KGRotatedLineSegmentSpaceBuilder::Attribute< string >( "name" ) +
        KGRotatedLineSegmentSpaceBuilder::Attribute< unsigned int >( "rotated_mesh_count" ) +
        KGRotatedLineSegmentSpaceBuilder::Attribute< unsigned int >( "flattened_mesh_count" ) +
        KGRotatedLineSegmentSpaceBuilder::Attribute< double >( "flattened_mesh_power" ) +
        KGRotatedLineSegmentSpaceBuilder::ComplexElement< KGPlanarLineSegment >( "line_segment" );

}
