#include "KGCutConeSurfaceBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

STATICINT sKGCutConeSurfaceBuilder = KGInterfaceBuilder::ComplexElement<KGCutConeSurface>("cut_cone_surface");

STATICINT sKGCutConeSurfaceBuilderStructure =
    KGCutConeSurfaceBuilder::Attribute<std::string>("name") +
    //! the first z coordinates in meters
    KGCutConeSurfaceBuilder::Attribute<double>("z1") +
    //! the first r coordinates in meters
    KGCutConeSurfaceBuilder::Attribute<double>("r1") +
    //! the second z coordinate in meters
    KGCutConeSurfaceBuilder::Attribute<double>("z2") +
    //! the second r coordinate in meters
    KGCutConeSurfaceBuilder::Attribute<double>("r2") +
    //! longitudinal mesh parameter (default is 1)
    KGCutConeSurfaceBuilder::Attribute<unsigned int>("longitudinal_mesh_count") +
    //! longitudinal meshing power (default is 1.)
    KGCutConeSurfaceBuilder::Attribute<double>("longitudinal_mesh_power") +
    //! axial mesh parameter (default is 16)
    KGCutConeSurfaceBuilder::Attribute<unsigned int>("axial_mesh_count");

}  // namespace katrin
