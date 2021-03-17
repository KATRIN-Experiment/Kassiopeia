#include "KGRotatedLineSegmentSurfaceBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

STATICINT sKGRotatedLineSegmentSurfaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGRotatedLineSegmentSurface>("rotated_line_segment_surface");

STATICINT sKGRotatedLineSegmentSurfaceBuilderStructure =
    KGRotatedLineSegmentSurfaceBuilder::Attribute<std::string>("name") +
    KGRotatedLineSegmentSurfaceBuilder::Attribute<unsigned int>("rotated_mesh_count") +
    KGRotatedLineSegmentSurfaceBuilder::ComplexElement<KGPlanarLineSegment>("line_segment");

}  // namespace katrin
