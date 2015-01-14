#include "KGExtrudedObjectBuilder.hh"
#include "KGInterfaceBuilder.hh"

namespace katrin
{

  static const int sKGExtrudedObjectLineBuilderStructure =
    KGExtrudedObjectLineBuilder::Attribute<double>("x1") +
    KGExtrudedObjectLineBuilder::Attribute<double>("y1") +
    KGExtrudedObjectLineBuilder::Attribute<double>("x2") +
    KGExtrudedObjectLineBuilder::Attribute<double>("y2");

  static const int sKGExtrudedObjectArcBuilderStructure =
    KGExtrudedObjectArcBuilder::Attribute<double>("x1") +
    KGExtrudedObjectArcBuilder::Attribute<double>("y1") +
    KGExtrudedObjectArcBuilder::Attribute<double>("x2") +
    KGExtrudedObjectArcBuilder::Attribute<double>("y2") +
    KGExtrudedObjectArcBuilder::Attribute<double>("radius") +
    KGExtrudedObjectArcBuilder::Attribute<bool>("positive_orientation");

  static const int sKGExtrudedObjectBuilderStructure =
    KGExtrudedObjectBuilder::Attribute<double>("z_min") +
    KGExtrudedObjectBuilder::Attribute<double>("z_max") +
    KGExtrudedObjectBuilder::Attribute<int>("longitudinal_mesh_count") +
    KGExtrudedObjectBuilder::Attribute<double>("longitudinal_mesh_power") +
    KGExtrudedObjectBuilder::Attribute<bool>("closed_form") +
    KGExtrudedObjectBuilder::ComplexElement<KGExtrudedObject::Line>("outer_line") +
    KGExtrudedObjectBuilder::ComplexElement<KGExtrudedObject::Line>("inner_line") +
    KGExtrudedObjectBuilder::ComplexElement<KGExtrudedObject::Arc>("outer_arc") +
    KGExtrudedObjectBuilder::ComplexElement<KGExtrudedObject::Arc>("inner_arc");

  static const int sKGExtrudedSurfaceBuilderStructure =
    KGExtrudedSurfaceBuilder::Attribute<string>("name") +
    KGExtrudedSurfaceBuilder::ComplexElement<KGExtrudedObject>("extruded_object");

  static const int sKGExtrudedSurfaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGWrappedSurface<KGExtrudedObject> >("extruded_surface");

  static const int sKGExtrudedSpaceBuilderStructure =
    KGExtrudedSpaceBuilder::Attribute<string>("name") +
    KGExtrudedSpaceBuilder::ComplexElement<KGExtrudedObject>("extruded_object");

  static const int sKGExtrudedSpaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGWrappedSpace<KGExtrudedObject> >("extruded_space");

}
