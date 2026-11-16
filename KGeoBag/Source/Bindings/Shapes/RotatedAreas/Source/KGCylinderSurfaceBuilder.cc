#include "KGCylinderSurfaceBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

STATICINT sKGCylinderSurfaceBuilder = KGInterfaceBuilder::ComplexElement<KGCylinderSurface>("cylinder_surface");

STATICINT sKGCylinderSurfaceBuilderStructure =
    KGCylinderSurfaceBuilder::Attribute<std::string>("name") +
    //! the first z position in meters
    KGCylinderSurfaceBuilder::Attribute<double>("z1") +
    //! the second z position in meters
    KGCylinderSurfaceBuilder::Attribute<double>("z2") + KGCylinderSurfaceBuilder::Attribute<double>("length") +
    //! radius in meters
    KGCylinderSurfaceBuilder::Attribute<double>("r") +
    //! longitudinal mesh parameter (default is 1)
    KGCylinderSurfaceBuilder::Attribute<unsigned int>("longitudinal_mesh_count") +
    //! longitudinal meshing power (default is 1.)
    KGCylinderSurfaceBuilder::Attribute<double>("longitudinal_mesh_power") +
    //! axial mesh parameter (default is 16)
    KGCylinderSurfaceBuilder::Attribute<unsigned int>("axial_mesh_count");

}  // namespace katrin
