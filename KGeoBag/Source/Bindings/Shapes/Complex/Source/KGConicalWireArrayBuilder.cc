#include "KGConicalWireArrayBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

STATICINT sKGConicalWireArrayBuilderStructure =
    KGConicalWireArrayBuilder::Attribute<double>("radius1") + KGConicalWireArrayBuilder::Attribute<double>("radius2") +
    KGConicalWireArrayBuilder::Attribute<double>("z1") + KGConicalWireArrayBuilder::Attribute<double>("z2") +
    KGConicalWireArrayBuilder::Attribute<unsigned int>("wire_count") +
    KGConicalWireArrayBuilder::Attribute<double>("theta_start") +
    KGConicalWireArrayBuilder::Attribute<double>("diameter") +
    KGConicalWireArrayBuilder::Attribute<unsigned int>("longitudinal_mesh_count") +
    KGConicalWireArrayBuilder::Attribute<double>("longitudinal_mesh_power");

STATICINT sKGConicalWireArraySurfaceBuilderStructure =
    KGConicalWireArraySurfaceBuilder::Attribute<string>("name") +
    KGConicalWireArraySurfaceBuilder::ComplexElement<KGConicalWireArray>("conical_wire_array");

STATICINT sKGConicalWireArraySurfaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGWrappedSurface<KGConicalWireArray>>("conical_wire_array_surface");

STATICINT sKGConicalWireArraySpaceBuilderStructure =
    KGConicalWireArraySpaceBuilder::Attribute<string>("name") +
    KGConicalWireArraySpaceBuilder::ComplexElement<KGConicalWireArray>("conical_wire_array");

STATICINT sKGConicalWireArraySpaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGWrappedSpace<KGConicalWireArray>>("conical_wire_array_space");

}  // namespace katrin
