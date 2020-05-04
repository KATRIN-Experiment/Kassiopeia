#include "KGPlanarLineSegmentBuilder.hh"

namespace katrin
{

STATICINT sKGPlanarLineSegmentBuilderStructure =
    KGPlanarLineSegmentBuilder::Attribute<double>("x1") + KGPlanarLineSegmentBuilder::Attribute<double>("y1") +
    KGPlanarLineSegmentBuilder::Attribute<double>("x2") + KGPlanarLineSegmentBuilder::Attribute<double>("y2") +
    KGPlanarLineSegmentBuilder::Attribute<unsigned int>("line_mesh_count") +
    KGPlanarLineSegmentBuilder::Attribute<double>("line_mesh_power");

}
