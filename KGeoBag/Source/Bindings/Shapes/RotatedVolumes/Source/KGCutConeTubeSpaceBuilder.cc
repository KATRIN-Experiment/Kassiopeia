#include "KGCutConeTubeSpaceBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

STATICINT sKGCutConeTubeSpaceBuilder = KGInterfaceBuilder::ComplexElement<KGCutConeTubeSpace>("cut_cone_tube_space");

STATICINT sKGCutConeTubeSpaceBuilderStructure =
    KGCutConeTubeSpaceBuilder::Attribute<std::string>("name") +
    //! the first side's z position in meters
    KGCutConeTubeSpaceBuilder::Attribute<double>("z1") +
    //! the second side's z position in meters
    KGCutConeTubeSpaceBuilder::Attribute<double>("z2") +
    //! one of the side's first radius in meters
    KGCutConeTubeSpaceBuilder::Attribute<double>("r11") +
    //! the other side's first radius in meters
    KGCutConeTubeSpaceBuilder::Attribute<double>("r12") +
    //! one of the side's second radius in meters
    KGCutConeTubeSpaceBuilder::Attribute<double>("r21") +
    //! the other side' second radius in meters
    KGCutConeTubeSpaceBuilder::Attribute<double>("r22") +
    //! radial mesh parameter (default is 1)
    KGCutConeTubeSpaceBuilder::Attribute<unsigned int>("radial_mesh_count") +
    //! radial meshing power (default is 1.)
    KGCutConeTubeSpaceBuilder::Attribute<double>("radial_mesh_power") +
    //! longitudinal mesh parameter (default is 1)
    KGCutConeTubeSpaceBuilder::Attribute<unsigned int>("longitudinal_mesh_count") +
    //! longitudinal meshing power (default is 1.)
    KGCutConeTubeSpaceBuilder::Attribute<double>("longitudinal_mesh_power") +
    //! axial mesh parameter (default is 16)
    KGCutConeTubeSpaceBuilder::Attribute<unsigned int>("axial_mesh_count");

}  // namespace katrin
