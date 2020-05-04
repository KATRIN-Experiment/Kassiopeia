#include "KGRodBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

STATICINT sKGRodVertexBuilderStructure = KGRodVertexBuilder::Attribute<double>("x") +
                                         KGRodVertexBuilder::Attribute<double>("y") +
                                         KGRodVertexBuilder::Attribute<double>("z");

STATICINT sKGRodBuilderStructure = KGRodBuilder::Attribute<string>("name") + KGRodBuilder::Attribute<double>("radius") +
                                   KGRodBuilder::Attribute<int>("longitudinal_mesh_count") +
                                   KGRodBuilder::Attribute<int>("axial_mesh_count") +
                                   KGRodBuilder::ComplexElement<KGRodVertex>("vertex");

STATICINT sKGRodSurfaceBuilderStructure =
    KGRodSurfaceBuilder::Attribute<string>("name") + KGRodSurfaceBuilder::ComplexElement<KGRod>("rod");

STATICINT sKGRodSurfaceBuilder = KGInterfaceBuilder::ComplexElement<KGWrappedSurface<KGRod>>("rod_surface");

STATICINT sKGRodSpaceBuilderStructure =
    KGRodSpaceBuilder::Attribute<string>("name") + KGRodSpaceBuilder::ComplexElement<KGRod>("rod");

STATICINT sKGRodSpaceBuilder = KGInterfaceBuilder::ComplexElement<KGWrappedSpace<KGRod>>("rod_space");

}  // namespace katrin
