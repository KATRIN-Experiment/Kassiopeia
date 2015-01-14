#include "KGConicalWireArrayBuilder.hh"
#include "KGInterfaceBuilder.hh"

namespace katrin
{

  static const int sKGConicalWireArrayBuilderStructure =
    KGConicalWireArrayBuilder::Attribute<double>("radius1") +
    KGConicalWireArrayBuilder::Attribute<double>("radius2") +
    KGConicalWireArrayBuilder::Attribute<double>("z1") +
    KGConicalWireArrayBuilder::Attribute<double>("z2") +
    KGConicalWireArrayBuilder::Attribute<int>("wire_count") +
    KGConicalWireArrayBuilder::Attribute<double>("theta_start") +
    KGConicalWireArrayBuilder::Attribute<double>("diameter") +
    KGConicalWireArrayBuilder::Attribute<int>("longitudinal_mesh_count");

  static const int sKGConicalWireArraySurfaceBuilderStructure =
    KGConicalWireArraySurfaceBuilder::Attribute<string>("name") +
    KGConicalWireArraySurfaceBuilder::ComplexElement<KGConicalWireArray>("conical_wire_array");

  static const int sKGConicalWireArraySurfaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGWrappedSurface<KGConicalWireArray> >("conical_wire_array_surface");

  static const int sKGConicalWireArraySpaceBuilderStructure =
    KGConicalWireArraySpaceBuilder::Attribute<string>("name") +
    KGConicalWireArraySpaceBuilder::ComplexElement<KGConicalWireArray>("conical_wire_array");

  static const int sKGConicalWireArraySpaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGWrappedSpace<KGConicalWireArray> >("conical_wire_array_space");

}
