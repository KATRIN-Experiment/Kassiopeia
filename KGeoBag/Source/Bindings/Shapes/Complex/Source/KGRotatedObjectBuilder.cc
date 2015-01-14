#include "KGRotatedObjectBuilder.hh"
#include "KGInterfaceBuilder.hh"

namespace katrin
{

  static const int sKGRotatedObjectLineBuilderStructure =
    KGRotatedObjectLineBuilder::Attribute<double>("z1") +
    KGRotatedObjectLineBuilder::Attribute<double>("r1") +
    KGRotatedObjectLineBuilder::Attribute<double>("z2") +
    KGRotatedObjectLineBuilder::Attribute<double>("r2");

  static const int sKGRotatedObjectArcBuilderStructure =
    KGRotatedObjectArcBuilder::Attribute<double>("z1") +
    KGRotatedObjectArcBuilder::Attribute<double>("r1") +
    KGRotatedObjectArcBuilder::Attribute<double>("z2") +
    KGRotatedObjectArcBuilder::Attribute<double>("r2") +
    KGRotatedObjectArcBuilder::Attribute<double>("radius") +
    KGRotatedObjectArcBuilder::Attribute<bool>("positive_orientation");

  static const int sKGRotatedObjectBuilderStructure =
    KGRotatedObjectBuilder::Attribute<int>("longitudinal_mesh_count_start") +
    KGRotatedObjectBuilder::Attribute<int>("longitudinal_mesh_count_end") +
    KGRotatedObjectBuilder::Attribute<int>("longitudinal_mesh_count") +
    KGRotatedObjectBuilder::Attribute<double>("longitudinal_mesh_power") +
    KGRotatedObjectBuilder::ComplexElement<KGRotatedObject::Line>("line") +
    KGRotatedObjectBuilder::ComplexElement<KGRotatedObject::Arc>("arc");

  static const int sKGRotatedSurfaceBuilderStructure =
    KGRotatedSurfaceBuilder::Attribute<string>("name") +
    KGRotatedSurfaceBuilder::ComplexElement<KGRotatedObject>("rotated_object");

  static const int sKGRotatedSurfaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGWrappedSurface<KGRotatedObject> >("rotated_surface");

  static const int sKGRotatedSpaceBuilderStructure =
    KGRotatedSpaceBuilder::Attribute<string>("name") +
    KGRotatedSpaceBuilder::ComplexElement<KGRotatedObject>("rotated_object");

  static const int sKGRotatedSpaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGWrappedSpace<KGRotatedObject> >("rotated_space");

}
