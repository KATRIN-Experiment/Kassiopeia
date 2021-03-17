#include "KGRotatedCircleSpaceBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

STATICINT sKGRotatedCircleSpaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGRotatedCircleSpace>("rotated_circle_space");

STATICINT sKGRotatedCircleSpaceBuilderStructure =
    KGRotatedCircleSpaceBuilder::Attribute<std::string>("name") +
    KGRotatedCircleSpaceBuilder::Attribute<unsigned int>("rotated_mesh_count") +
    KGRotatedCircleSpaceBuilder::ComplexElement<KGPlanarCircle>("circle");

}  // namespace katrin
