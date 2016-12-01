#include "KGExtrudedArcSegmentSurfaceBuilder.hh"
#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

    STATICINT sKGExtrudedArcSegmentSurfaceBuilder =
        KGInterfaceBuilder::ComplexElement< KGExtrudedArcSegmentSurface >( "extruded_arc_segment_surface" );

    STATICINT sKGExtrudedArcSegmentSurfaceBuilderStructure =
        KGExtrudedArcSegmentSurfaceBuilder::Attribute< string >( "name" ) +
        KGExtrudedArcSegmentSurfaceBuilder::Attribute< double >( "zmin" ) +
        KGExtrudedArcSegmentSurfaceBuilder::Attribute< double >( "zmax" ) +
        KGExtrudedArcSegmentSurfaceBuilder::Attribute< unsigned int >( "extruded_mesh_count" ) +
        KGExtrudedArcSegmentSurfaceBuilder::Attribute< double >( "extruded_mesh_power" ) +
        KGExtrudedArcSegmentSurfaceBuilder::ComplexElement< KGPlanarArcSegment >( "arc_segment" );

}
