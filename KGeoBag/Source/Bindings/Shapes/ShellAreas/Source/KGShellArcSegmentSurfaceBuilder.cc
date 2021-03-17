#include "KGShellArcSegmentSurfaceBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

STATICINT sKGShellArcSegmentSurfaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGShellArcSegmentSurface>("shell_arc_segment_surface");

STATICINT sKGShellArcSegmentSurfaceBuilderStructure =
    KGShellArcSegmentSurfaceBuilder::Attribute<std::string>("name") +
    KGShellArcSegmentSurfaceBuilder::Attribute<double>("angle_start") +
    KGShellArcSegmentSurfaceBuilder::Attribute<double>("angle_stop") +
    KGShellArcSegmentSurfaceBuilder::Attribute<unsigned int>("shell_mesh_count") +
    KGShellArcSegmentSurfaceBuilder::Attribute<double>("shell_mesh_power") +
    KGShellArcSegmentSurfaceBuilder::ComplexElement<KGPlanarArcSegment>("arc_segment");

}  // namespace katrin
