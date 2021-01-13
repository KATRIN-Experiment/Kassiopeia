#include "KGShellCircleSurfaceBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

STATICINT sKGShellCircleSurfaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGShellCircleSurface>("shell_circle_surface");

STATICINT sKGShellCircleSurfaceBuilderStructure =
    KGShellCircleSurfaceBuilder::Attribute<std::string>("name") +
    KGShellCircleSurfaceBuilder::Attribute<double>("angle_start") +
    KGShellCircleSurfaceBuilder::Attribute<double>("angle_stop") +
    KGShellCircleSurfaceBuilder::Attribute<unsigned int>("shell_mesh_count") +
    KGShellCircleSurfaceBuilder::Attribute<double>("shell_mesh_power") +
    KGShellCircleSurfaceBuilder::ComplexElement<KGPlanarCircle>("circle");

}  // namespace katrin
