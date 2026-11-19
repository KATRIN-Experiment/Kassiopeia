#include "KGCylinderTubeSpaceBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

STATICINT sKGCylinderTubeSpaceBuilder = KGInterfaceBuilder::ComplexElement<KGCylinderTubeSpace>("cylinder_tube_space");

STATICINT sKGCylinderTubeSpaceBuilderStructure =
    KGCylinderTubeSpaceBuilder::Attribute<std::string>("name") +
    //! the first z position in meters
    KGCylinderTubeSpaceBuilder::Attribute<double>("z1") +
    //! the second z position in meters
    KGCylinderTubeSpaceBuilder::Attribute<double>("z2") + KGCylinderTubeSpaceBuilder::Attribute<double>("length") +
    //! the first radius in meters
    KGCylinderTubeSpaceBuilder::Attribute<double>("r1") +
    //! the second radius in meters
    KGCylinderTubeSpaceBuilder::Attribute<double>("r2") +
    //! longitudinal mesh parameter (default is 1)
    KGCylinderTubeSpaceBuilder::Attribute<unsigned int>("longitudinal_mesh_count") +
    //! longitudinal meshing power (default is 1.)
    KGCylinderTubeSpaceBuilder::Attribute<double>("longitudinal_mesh_power") +
    //! radial mesh parameter (default is 1)
    KGCylinderTubeSpaceBuilder::Attribute<unsigned int>("radial_mesh_count") +
    //! radial meshing power (default is 1.)
    KGCylinderTubeSpaceBuilder::Attribute<double>("radial_mesh_power") +
    //! axial mesh parameter (default is 16)
    KGCylinderTubeSpaceBuilder::Attribute<unsigned int>("axial_mesh_count");

}  // namespace katrin
