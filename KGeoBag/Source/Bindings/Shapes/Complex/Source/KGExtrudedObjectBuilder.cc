#include "KGExtrudedObjectBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

STATICINT sKGExtrudedObjectLineBuilderStructure =
    KGExtrudedObjectLineBuilder::Attribute<double>("x1") + KGExtrudedObjectLineBuilder::Attribute<double>("y1") +
    KGExtrudedObjectLineBuilder::Attribute<double>("x2") + KGExtrudedObjectLineBuilder::Attribute<double>("y2");

STATICINT sKGExtrudedObjectArcBuilderStructure =
    KGExtrudedObjectArcBuilder::Attribute<double>("x1") + KGExtrudedObjectArcBuilder::Attribute<double>("y1") +
    KGExtrudedObjectArcBuilder::Attribute<double>("x2") + KGExtrudedObjectArcBuilder::Attribute<double>("y2") +
    KGExtrudedObjectArcBuilder::Attribute<double>("radius") +
    KGExtrudedObjectArcBuilder::Attribute<bool>("positive_orientation");

STATICINT sKGExtrudedObjectBuilderStructure =
    KGExtrudedObjectBuilder::Attribute<double>("z_min") + KGExtrudedObjectBuilder::Attribute<double>("z_max") +
    KGExtrudedObjectBuilder::Attribute<int>("longitudinal_mesh_count") +
    KGExtrudedObjectBuilder::Attribute<double>("longitudinal_mesh_power") +
    KGExtrudedObjectBuilder::Attribute<bool>("closed_form") +
    KGExtrudedObjectBuilder::ComplexElement<KGExtrudedObject::Line>("outer_line") +
    KGExtrudedObjectBuilder::ComplexElement<KGExtrudedObject::Line>("inner_line") +
    KGExtrudedObjectBuilder::ComplexElement<KGExtrudedObject::Arc>("outer_arc") +
    KGExtrudedObjectBuilder::ComplexElement<KGExtrudedObject::Arc>("inner_arc");

STATICINT sKGExtrudedSurfaceBuilderStructure =
    KGExtrudedSurfaceBuilder::Attribute<std::string>("name") +
    KGExtrudedSurfaceBuilder::ComplexElement<KGExtrudedObject>("extruded_object");

STATICINT sKGExtrudedSurfaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGWrappedSurface<KGExtrudedObject>>("extruded_surface");

STATICINT sKGExtrudedSpaceBuilderStructure =
    KGExtrudedSpaceBuilder::Attribute<std::string>("name") +
    KGExtrudedSpaceBuilder::ComplexElement<KGExtrudedObject>("extruded_object");

STATICINT sKGExtrudedSpaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGWrappedSpace<KGExtrudedObject>>("extruded_space");

}  // namespace katrin
