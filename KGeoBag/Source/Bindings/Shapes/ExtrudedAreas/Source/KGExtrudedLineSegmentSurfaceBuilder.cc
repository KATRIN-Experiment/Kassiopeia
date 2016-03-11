#include "KGExtrudedLineSegmentSurfaceBuilder.hh"
#include "KGInterfaceBuilder.hh"

namespace katrin
{

    STATICINT sKGExtrudedLineSegmentSurfaceBuilder =
        KGInterfaceBuilder::ComplexElement< KGExtrudedLineSegmentSurface >( "extruded_line_segment_surface" );

    STATICINT sKGExtrudedLineSegmentSurfaceBuilderStructure =
        KGExtrudedLineSegmentSurfaceBuilder::Attribute< string >( "name" ) +
        KGExtrudedLineSegmentSurfaceBuilder::Attribute< double >( "zmin" ) +
        KGExtrudedLineSegmentSurfaceBuilder::Attribute< double >( "zmax" ) +
        KGExtrudedLineSegmentSurfaceBuilder::Attribute< unsigned int >( "extruded_mesh_count" ) +
        KGExtrudedLineSegmentSurfaceBuilder::Attribute< double >( "extruded_mesh_power" ) +
        KGExtrudedLineSegmentSurfaceBuilder::ComplexElement< KGPlanarLineSegment >( "line_segment" );

}
