#include "KGPlanarArcSegmentBuilder.hh"

namespace katrin
{

    static const int sKGPlanarArcSegmentBuilderStructure =
        KGPlanarArcSegmentBuilder::Attribute< double >( "x1" ) +
        KGPlanarArcSegmentBuilder::Attribute< double >( "y1" ) +
        KGPlanarArcSegmentBuilder::Attribute< double >( "x2" ) +
        KGPlanarArcSegmentBuilder::Attribute< double >( "y2" ) +
        KGPlanarArcSegmentBuilder::Attribute< double >( "radius" ) +
        KGPlanarArcSegmentBuilder::Attribute< bool >( "right" ) +
        KGPlanarArcSegmentBuilder::Attribute< bool >( "short" ) +
        KGPlanarArcSegmentBuilder::Attribute< unsigned int >( "arc_mesh_count" );

}
