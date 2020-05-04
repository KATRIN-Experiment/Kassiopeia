#include "KGCylinderTubeSpaceBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

STATICINT sKGCylinderTubeSpaceBuilder = KGInterfaceBuilder::ComplexElement<KGCylinderTubeSpace>("cylinder_tube_space");

STATICINT sKGCylinderTubeSpaceBuilderStructure =
    KGCylinderTubeSpaceBuilder::Attribute<string>("name") + KGCylinderTubeSpaceBuilder::Attribute<double>("z1") +
    KGCylinderTubeSpaceBuilder::Attribute<double>("z2") + KGCylinderTubeSpaceBuilder::Attribute<double>("length") +
    KGCylinderTubeSpaceBuilder::Attribute<double>("r1") + KGCylinderTubeSpaceBuilder::Attribute<double>("r2") +
    KGCylinderTubeSpaceBuilder::Attribute<unsigned int>("longitudinal_mesh_count") +
    KGCylinderTubeSpaceBuilder::Attribute<double>("longitudinal_mesh_power") +
    KGCylinderTubeSpaceBuilder::Attribute<unsigned int>("radial_mesh_count") +
    KGCylinderTubeSpaceBuilder::Attribute<double>("radial_mesh_power") +
    KGCylinderTubeSpaceBuilder::Attribute<unsigned int>("axial_mesh_count");

}  // namespace katrin
