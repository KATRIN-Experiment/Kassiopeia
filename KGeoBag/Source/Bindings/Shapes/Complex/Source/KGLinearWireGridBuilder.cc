#include "KGLinearWireGridBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

STATICINT sKGLinearWireGridBuilderStructure =
    KGLinearWireGridBuilder::Attribute<double>("radius") + KGLinearWireGridBuilder::Attribute<double>("pitch") +
    KGLinearWireGridBuilder::Attribute<double>("diameter") +
    KGLinearWireGridBuilder::Attribute<unsigned int>("longitudinal_mesh_count") +
    KGLinearWireGridBuilder::Attribute<double>("longitudinal_mesh_power") +
    KGLinearWireGridBuilder::Attribute<bool>("add_outer_circle");

STATICINT sKGLinearWireGridSurfaceBuilderStructure =
    KGLinearWireGridSurfaceBuilder::Attribute<string>("name") +
    KGLinearWireGridSurfaceBuilder::ComplexElement<KGLinearWireGrid>("linear_wire_grid");

STATICINT sKGLinearWireGridSurfaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGWrappedSurface<KGLinearWireGrid>>("linear_wire_grid_surface");

STATICINT sKGLinearWireGridSpaceBuilderStructure =
    KGLinearWireGridSpaceBuilder::Attribute<string>("name") +
    KGLinearWireGridSpaceBuilder::ComplexElement<KGLinearWireGrid>("linear_wire_grid");

STATICINT sKGLinearWireGridSpaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGWrappedSpace<KGLinearWireGrid>>("linear_wire_grid_space");

}  // namespace katrin
