#include "KGConeSpaceBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

STATICINT sKGConeSpaceBuilder = KGInterfaceBuilder::ComplexElement<KGConeSpace>("cone_space");

STATICINT sKGConeSpaceBuilderStructure = KGConeSpaceBuilder::Attribute<std::string>("name") +
                                         //! apex z position in meters
                                         KGConeSpaceBuilder::Attribute<double>("za") +
                                         //! base z position in meters
                                         KGConeSpaceBuilder::Attribute<double>("zb") +
                                         //! base radius in meters
                                         KGConeSpaceBuilder::Attribute<double>("rb") +
                                         //! longitudinal mesh parameter (default is 1)
                                         KGConeSpaceBuilder::Attribute<unsigned int>("longitudinal_mesh_count") +
                                         //! longitudinal meshing power (default is 1.)
                                         KGConeSpaceBuilder::Attribute<double>("longitudinal_mesh_power") +
                                         //! radial mesh parameter (default is 1)
                                         KGConeSpaceBuilder::Attribute<unsigned int>("radial_mesh_count") +
                                         //! radial meshing power (default is 1.)
                                         KGConeSpaceBuilder::Attribute<double>("radial_mesh_power") +
                                         //! axial mesh parameter (default is 16)
                                         KGConeSpaceBuilder::Attribute<unsigned int>("axial_mesh_count");

}  // namespace katrin
