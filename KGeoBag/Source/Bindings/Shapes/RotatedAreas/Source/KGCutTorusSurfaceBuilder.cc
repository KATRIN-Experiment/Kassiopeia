#include "KGCutTorusSurfaceBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

STATICINT sKGCutTorusSurfaceBuilder = KGInterfaceBuilder::ComplexElement<KGCutTorusSurface>("cut_torus_surface");

STATICINT sKGCutTorusSurfaceBuilderStructure =
    KGCutTorusSurfaceBuilder::Attribute<std::string>("name") +
    //! the first z coordinate in meters
    KGCutTorusSurfaceBuilder::Attribute<double>("z1") +
    //! the first r coordinate in meters
    KGCutTorusSurfaceBuilder::Attribute<double>("r1") +
    //! the second z coordinate in meters
    KGCutTorusSurfaceBuilder::Attribute<double>("z2") +
    //! the second r coordinate in meters
    KGCutTorusSurfaceBuilder::Attribute<double>("r2") +
    //! the toroidal radius in meters
    KGCutTorusSurfaceBuilder::Attribute<double>("radius") +
    //! is the arc on the right side of the directed line connecting point 1 to point 2?
    KGCutTorusSurfaceBuilder::Attribute<bool>("right") +
    //! does the arc subtend less than pi radians?
    KGCutTorusSurfaceBuilder::Attribute<bool>("short") +
    //! toroidal mesh parameter (default is 64)
    KGCutTorusSurfaceBuilder::Attribute<unsigned int>("toroidal_mesh_count") +
    //! axial mesh parameter (default is 64)
    KGCutTorusSurfaceBuilder::Attribute<unsigned int>("axial_mesh_count");

}  // namespace katrin
