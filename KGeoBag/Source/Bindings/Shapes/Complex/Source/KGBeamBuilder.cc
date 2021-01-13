#include "KGBeamBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

STATICINT sKGBeamLineBuilderStructure =
    KGBeamLineBuilder::Attribute<double>("x1") + KGBeamLineBuilder::Attribute<double>("y1") +
    KGBeamLineBuilder::Attribute<double>("z1") + KGBeamLineBuilder::Attribute<double>("x2") +
    KGBeamLineBuilder::Attribute<double>("y2") + KGBeamLineBuilder::Attribute<double>("z2");

STATICINT sKGBeamBuilderStructure =
    KGBeamBuilder::Attribute<int>("longitudinal_mesh_count") + KGBeamBuilder::Attribute<int>("axial_mesh_count") +
    KGBeamBuilder::ComplexElement<KGBeamLine>("start_line") + KGBeamBuilder::ComplexElement<KGBeamLine>("end_line");

STATICINT sKGBeamSurfaceBuilderStructure =
    KGBeamSurfaceBuilder::Attribute<std::string>("name") + KGBeamSurfaceBuilder::ComplexElement<KGBeam>("beam");

STATICINT sKGBeamSurfaceBuilder = KGInterfaceBuilder::ComplexElement<KGWrappedSurface<KGBeam>>("beam_surface");

STATICINT sKGBeamSpaceBuilderStructure =
    KGBeamSpaceBuilder::Attribute<std::string>("name") + KGBeamSpaceBuilder::ComplexElement<KGBeam>("beam");

STATICINT sKGBeamSpaceBuilder = KGInterfaceBuilder::ComplexElement<KGWrappedSpace<KGBeam>>("beam_space");

}  // namespace katrin
