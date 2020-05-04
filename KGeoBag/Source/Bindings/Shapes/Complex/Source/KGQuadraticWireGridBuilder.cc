#include "KGQuadraticWireGridBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

STATICINT sKGQuadraticWireGridBuilderStructure =
    KGQuadraticWireGridBuilder::Attribute<double>("radius") + KGQuadraticWireGridBuilder::Attribute<double>("pitch") +
    KGQuadraticWireGridBuilder::Attribute<double>("diameter") +
    KGQuadraticWireGridBuilder::Attribute<unsigned int>("mesh_count_per_pitch") +
    KGQuadraticWireGridBuilder::Attribute<bool>("add_outer_circle");

STATICINT sKGQuadraticWireGridSurfaceBuilderStructure =
    KGQuadraticWireGridSurfaceBuilder::Attribute<string>("name") +
    KGQuadraticWireGridSurfaceBuilder::ComplexElement<KGQuadraticWireGrid>("quadratic_wire_grid");

STATICINT sKGQuadraticWireGridSurfaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGWrappedSurface<KGQuadraticWireGrid>>("quadratic_wire_grid_surface");

STATICINT sKGQuadraticWireGridSpaceBuilderStructure =
    KGQuadraticWireGridSpaceBuilder::Attribute<string>("name") +
    KGQuadraticWireGridSpaceBuilder::ComplexElement<KGQuadraticWireGrid>("quadratic_wire_grid");

STATICINT sKGQuadraticWireGridSpaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGWrappedSpace<KGQuadraticWireGrid>>("quadratic_wire_grid_space");

}  // namespace katrin
