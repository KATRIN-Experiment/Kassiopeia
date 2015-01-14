#include "KGRodBuilder.hh"
#include "KGInterfaceBuilder.hh"

namespace katrin
{

  static const int sKGRodVertexBuilderStructure =
    KGRodVertexBuilder::Attribute<double>("x") +
    KGRodVertexBuilder::Attribute<double>("y") +
    KGRodVertexBuilder::Attribute<double>("z");

  static const int sKGRodBuilderStructure =
    KGRodBuilder::Attribute<double>("radius") +
    KGRodBuilder::Attribute<int>("longitudinal_mesh_count") +
    KGRodBuilder::Attribute<int>("axial_mesh_count") +
    KGRodBuilder::ComplexElement<KGRodVertex>("vertex");

  static const int sKGRodSurfaceBuilderStructure =
    KGRodSurfaceBuilder::Attribute<string>("name") +
    KGRodSurfaceBuilder::ComplexElement<KGRod>("rod");

  static const int sKGRodSurfaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGWrappedSurface<KGRod> >("rod_surface");

  static const int sKGRodSpaceBuilderStructure =
    KGRodSpaceBuilder::Attribute<string>("name") +
    KGRodSpaceBuilder::ComplexElement<KGRod>("rod");

  static const int sKGRodSpaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGWrappedSpace<KGRod> >("rod_space");

}
