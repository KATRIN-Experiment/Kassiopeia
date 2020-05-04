#include "KGRotatedArcSegmentSpaceBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

STATICINT sKGRotatedArcSegmentSpaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGRotatedArcSegmentSpace>("rotated_arc_segment_space");

STATICINT sKGRotatedArcSegmentSpaceBuilderStructure =
    KGRotatedArcSegmentSpaceBuilder::Attribute<string>("name") +
    KGRotatedArcSegmentSpaceBuilder::Attribute<unsigned int>("rotated_mesh_count") +
    KGRotatedArcSegmentSpaceBuilder::Attribute<unsigned int>("flattened_mesh_count") +
    KGRotatedArcSegmentSpaceBuilder::Attribute<double>("flattened_mesh_power") +
    KGRotatedArcSegmentSpaceBuilder::ComplexElement<KGPlanarArcSegment>("arc_segment");

}  // namespace katrin
