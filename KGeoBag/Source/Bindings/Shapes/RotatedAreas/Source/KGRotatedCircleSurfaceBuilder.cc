#include "KGRotatedCircleSurfaceBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

STATICINT sKGRotatedCircleSurfaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGRotatedCircleSurface>("rotated_circle_surface");

STATICINT sKGRotatedCircleSurfaceBuilderStructure =
    KGRotatedCircleSurfaceBuilder::Attribute<std::string>("name") +
    KGRotatedCircleSurfaceBuilder::Attribute<unsigned int>("rotated_mesh_count") +
    KGRotatedCircleSurfaceBuilder::ComplexElement<KGPlanarCircle>("circle");

}  // namespace katrin
