#include "KGExtrudedArcSegmentSurfaceBuilder.hh"
#include "KGInterfaceBuilder.hh"

namespace katrin
{

    static const int sKGExtrudedArcSegmentSurfaceBuilder =
        KGInterfaceBuilder::ComplexElement< KGExtrudedArcSegmentSurface >( "extruded_arc_segment_surface" );

    static const int sKGExtrudedArcSegmentSurfaceBuilderStructure =
        KGExtrudedArcSegmentSurfaceBuilder::Attribute< string >( "name" ) +
        KGExtrudedArcSegmentSurfaceBuilder::Attribute< double >( "zmin" ) +
        KGExtrudedArcSegmentSurfaceBuilder::Attribute< double >( "zmax" ) +
        KGExtrudedArcSegmentSurfaceBuilder::Attribute< unsigned int >( "extruded_mesh_count" ) +
        KGExtrudedArcSegmentSurfaceBuilder::Attribute< double >( "extruded_mesh_power" ) +
        KGExtrudedArcSegmentSurfaceBuilder::ComplexElement< KGPlanarArcSegment >( "arc_segment" );

}
