#include "KGCutConeSpaceBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

STATICINT sKGCutConeSpaceBuilder = KGInterfaceBuilder::ComplexElement<KGCutConeSpace>("cut_cone_space");

STATICINT sKGCutConeSpaceBuilderStructure =
    KGCutConeSpaceBuilder::Attribute<std::string>("name") + KGCutConeSpaceBuilder::Attribute<double>("z1") +
    KGCutConeSpaceBuilder::Attribute<double>("z2") + KGCutConeSpaceBuilder::Attribute<double>("r1") +
    KGCutConeSpaceBuilder::Attribute<double>("r2") +
    KGCutConeSpaceBuilder::Attribute<unsigned int>("longitudinal_mesh_count") +
    KGCutConeSpaceBuilder::Attribute<double>("longitudinal_mesh_power") +
    KGCutConeSpaceBuilder::Attribute<unsigned int>("radial_mesh_count") +
    KGCutConeSpaceBuilder::Attribute<double>("radial_mesh_power") +
    KGCutConeSpaceBuilder::Attribute<unsigned int>("axial_mesh_count");
}  // namespace katrin
