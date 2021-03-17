#include "KGTorusSurfaceBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

STATICINT sKGTorusSurfaceBuilder = KGInterfaceBuilder::ComplexElement<KGTorusSurface>("torus_surface");

STATICINT sKGTorusSurfaceBuilderStructure =
    KGTorusSurfaceBuilder::Attribute<std::string>("name") + KGTorusSurfaceBuilder::Attribute<double>("z") +
    KGTorusSurfaceBuilder::Attribute<double>("r") + KGTorusSurfaceBuilder::Attribute<double>("radius") +
    KGTorusSurfaceBuilder::Attribute<unsigned int>("toroidal_mesh_count") +
    KGTorusSurfaceBuilder::Attribute<unsigned int>("axial_mesh_count");

}  // namespace katrin
