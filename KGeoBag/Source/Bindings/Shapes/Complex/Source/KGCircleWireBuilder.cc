#include "KGCircleWireBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

STATICINT sKGCircleWireBuilderStructure = KGCircleWireBuilder::Attribute<double>("radius") +
                                          KGCircleWireBuilder::Attribute<double>("diameter") +
                                          KGCircleWireBuilder::Attribute<unsigned int>("mesh_count");

STATICINT sKGCircleWireSurfaceBuilderStructure =
    KGCircleWireSurfaceBuilder::Attribute<std::string>("name") +
    KGCircleWireSurfaceBuilder::ComplexElement<KGCircleWire>("circle_wire");

STATICINT sKGCircleWireSurfaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGWrappedSurface<KGCircleWire>>("circle_wire_surface");

STATICINT sKGCircleWireSpaceBuilderStructure = KGCircleWireSpaceBuilder::Attribute<std::string>("name") +
                                               KGCircleWireSpaceBuilder::ComplexElement<KGCircleWire>("circle_wire");

STATICINT sKGCircleWireSpaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGWrappedSpace<KGCircleWire>>("circle_wire_space");

}  // namespace katrin
