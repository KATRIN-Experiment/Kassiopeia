#include "KGExtrudedPolyLineSurfaceBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

STATICINT sKGExtrudedPolyLineSurfaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGExtrudedPolyLineSurface>("extruded_poly_line_surface");

STATICINT sKGExtrudedPolyLineSurfaceBuilderStructure =
    KGExtrudedPolyLineSurfaceBuilder::Attribute<std::string>("name") +
    KGExtrudedPolyLineSurfaceBuilder::Attribute<double>("zmin") +
    KGExtrudedPolyLineSurfaceBuilder::Attribute<double>("zmax") +
    KGExtrudedPolyLineSurfaceBuilder::Attribute<unsigned int>("extruded_mesh_count") +
    KGExtrudedPolyLineSurfaceBuilder::Attribute<double>("extruded_mesh_power") +
    KGExtrudedPolyLineSurfaceBuilder::ComplexElement<KGPlanarPolyLine>("poly_line");

}  // namespace katrin
