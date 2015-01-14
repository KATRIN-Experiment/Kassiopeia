#include "KGBeamBuilder.hh"
#include "KGInterfaceBuilder.hh"

namespace katrin
{

  static const int sKGBeamLineBuilderStructure =
    KGBeamLineBuilder::Attribute<double>("x1") +
    KGBeamLineBuilder::Attribute<double>("y1") +
    KGBeamLineBuilder::Attribute<double>("z1") +
    KGBeamLineBuilder::Attribute<double>("x2") +
    KGBeamLineBuilder::Attribute<double>("y2") +
    KGBeamLineBuilder::Attribute<double>("z2");

  static const int sKGBeamBuilderStructure =
    KGBeamBuilder::Attribute<int>("longitudinal_mesh_count") +
    KGBeamBuilder::Attribute<int>("axial_mesh_count") +
    KGBeamBuilder::ComplexElement<KGBeamLine>("start_line") +
    KGBeamBuilder::ComplexElement<KGBeamLine>("end_line");

  static const int sKGBeamSurfaceBuilderStructure =
    KGBeamSurfaceBuilder::Attribute<string>("name") +
    KGBeamSurfaceBuilder::ComplexElement<KGBeam>("beam");

  static const int sKGBeamSurfaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGWrappedSurface<KGBeam> >("beam_surface");

  static const int sKGBeamSpaceBuilderStructure =
    KGBeamSpaceBuilder::Attribute<string>("name") +
    KGBeamSpaceBuilder::ComplexElement<KGBeam>("beam");

  static const int sKGBeamSpaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGWrappedSpace<KGBeam> >("beam_space");

}
