#include "KGConeSurfaceBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

STATICINT sKGConeSurfaceBuilder = KGInterfaceBuilder::ComplexElement<KGConeSurface>("cone_surface");

STATICINT sKGConeSurfaceBuilderStructure = KGConeSurfaceBuilder::Attribute<std::string>("name") +
                                           //! apex z position in meters
                                           KGConeSurfaceBuilder::Attribute<double>("za") +
                                           //! base z position in meters
                                           KGConeSurfaceBuilder::Attribute<double>("zb") +
                                           //! base radius in meters
                                           KGConeSurfaceBuilder::Attribute<double>("rb") +
                                           //! longitudinal mesh parameter (default is 1)
                                           KGConeSurfaceBuilder::Attribute<unsigned int>("longitudinal_mesh_count") +
                                           //! longitudinal meshing power (default is 1.)
                                           KGConeSurfaceBuilder::Attribute<double>("longitudinal_mesh_power") +
                                           //! axial mesh parameter (default is 16)
                                           KGConeSurfaceBuilder::Attribute<unsigned int>("axial_mesh_count");

}  // namespace katrin
