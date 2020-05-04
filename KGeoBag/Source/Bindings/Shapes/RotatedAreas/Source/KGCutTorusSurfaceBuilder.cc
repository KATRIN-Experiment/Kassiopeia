#include "KGCutTorusSurfaceBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

STATICINT sKGCutTorusSurfaceBuilder = KGInterfaceBuilder::ComplexElement<KGCutTorusSurface>("cut_torus_surface");

STATICINT sKGCutTorusSurfaceBuilderStructure =
    KGCutTorusSurfaceBuilder::Attribute<string>("name") + KGCutTorusSurfaceBuilder::Attribute<double>("z1") +
    KGCutTorusSurfaceBuilder::Attribute<double>("r1") + KGCutTorusSurfaceBuilder::Attribute<double>("z2") +
    KGCutTorusSurfaceBuilder::Attribute<double>("r2") + KGCutTorusSurfaceBuilder::Attribute<double>("radius") +
    KGCutTorusSurfaceBuilder::Attribute<bool>("right") + KGCutTorusSurfaceBuilder::Attribute<bool>("short") +
    KGCutTorusSurfaceBuilder::Attribute<unsigned int>("toroidal_mesh_count") +
    KGCutTorusSurfaceBuilder::Attribute<unsigned int>("axial_mesh_count");

}  // namespace katrin
