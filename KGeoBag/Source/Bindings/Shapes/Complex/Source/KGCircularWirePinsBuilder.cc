#include "KGCircularWirePinsBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

STATICINT sKGCircularWirePinsBuilderStructure = KGCircularWirePinsBuilder::Attribute<double>("inner_radius") +
                                                KGCircularWirePinsBuilder::Attribute<double>("outer_radius") +
                                                KGCircularWirePinsBuilder::Attribute<unsigned int>("n_pins") +
                                                KGCircularWirePinsBuilder::Attribute<double>("diameter") +
                                                KGCircularWirePinsBuilder::Attribute<double>("rotation_angle") +
                                                KGCircularWirePinsBuilder::Attribute<unsigned int>("mesh_count") +
                                                KGCircularWirePinsBuilder::Attribute<double>("mesh_power");

STATICINT sKGCircularWirePinsSurfaceBuilderStructure =
    KGCircularWirePinsSurfaceBuilder::Attribute<string>("name") +
    KGCircularWirePinsSurfaceBuilder::ComplexElement<KGCircularWirePins>("circular_wire_pins");

STATICINT sKGCircularWirePinsSurfaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGWrappedSurface<KGCircularWirePins>>("circular_wire_pins_surface");

STATICINT sKGCircleWireSpaceBuilderStructure =
    KGCircularWirePinsSpaceBuilder::Attribute<string>("name") +
    KGCircularWirePinsSpaceBuilder::ComplexElement<KGCircularWirePins>("circular_wire_pins");

STATICINT sKGCircularWirePinsSpaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGWrappedSpace<KGCircularWirePins>>("circular_wire_pins_space");

}  // namespace katrin
