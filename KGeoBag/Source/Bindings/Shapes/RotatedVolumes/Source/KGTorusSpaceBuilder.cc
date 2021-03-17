#include "KGTorusSpaceBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

STATICINT sKGTorusSpaceBuilder = KGInterfaceBuilder::ComplexElement<KGTorusSpace>("torus_space");

STATICINT sKGTorusSpaceBuilderStructure =
    KGTorusSpaceBuilder::Attribute<std::string>("name") + KGTorusSpaceBuilder::Attribute<double>("z") +
    KGTorusSpaceBuilder::Attribute<double>("r") + KGTorusSpaceBuilder::Attribute<double>("radius") +
    KGTorusSpaceBuilder::Attribute<unsigned int>("toroidal_mesh_count") +
    KGTorusSpaceBuilder::Attribute<unsigned int>("axial_mesh_count");

}  // namespace katrin
