#include "KGRotatedLineSegmentSpaceBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

STATICINT sKGRotatedLineSegmentSpaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGRotatedLineSegmentSpace>("rotated_line_segment_space");

STATICINT sKGRotatedLineSegmentSpaceBuilderStructure =
    KGRotatedLineSegmentSpaceBuilder::Attribute<std::string>("name") +
    KGRotatedLineSegmentSpaceBuilder::Attribute<unsigned int>("rotated_mesh_count") +
    KGRotatedLineSegmentSpaceBuilder::Attribute<unsigned int>("flattened_mesh_count") +
    KGRotatedLineSegmentSpaceBuilder::Attribute<double>("flattened_mesh_power") +
    KGRotatedLineSegmentSpaceBuilder::ComplexElement<KGPlanarLineSegment>("line_segment");

}  // namespace katrin
