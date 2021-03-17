#include "KGExtrudedPolyLoopSpaceBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

STATICINT sKGExtrudedPolyLoopSpaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGExtrudedPolyLoopSpace>("extruded_poly_loop_space");

STATICINT sKGExtrudedPolyLoopSpaceBuilderStructure =
    KGExtrudedPolyLoopSpaceBuilder::Attribute<std::string>("name") +
    KGExtrudedPolyLoopSpaceBuilder::Attribute<double>("zmin") +
    KGExtrudedPolyLoopSpaceBuilder::Attribute<double>("zmax") +
    KGExtrudedPolyLoopSpaceBuilder::Attribute<unsigned int>("extruded_mesh_count") +
    KGExtrudedPolyLoopSpaceBuilder::Attribute<double>("extruded_mesh_power") +
    KGExtrudedPolyLoopSpaceBuilder::Attribute<unsigned int>("flattened_mesh_count") +
    KGExtrudedPolyLoopSpaceBuilder::Attribute<double>("flattened_mesh_power") +
    KGExtrudedPolyLoopSpaceBuilder::ComplexElement<KGPlanarPolyLoop>("poly_loop");

}  // namespace katrin
