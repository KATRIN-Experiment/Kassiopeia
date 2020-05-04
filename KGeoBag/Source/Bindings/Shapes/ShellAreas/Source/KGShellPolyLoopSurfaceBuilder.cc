#include "KGShellPolyLoopSurfaceBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

STATICINT sKGShellPolyLoopSurfaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGShellPolyLoopSurface>("shell_poly_loop_surface");

STATICINT sKGShellPolyLoopSurfaceBuilderStructure =
    KGShellPolyLoopSurfaceBuilder::Attribute<string>("name") +
    KGShellPolyLoopSurfaceBuilder::Attribute<double>("angle_start") +
    KGShellPolyLoopSurfaceBuilder::Attribute<double>("angle_stop") +
    KGShellPolyLoopSurfaceBuilder::Attribute<unsigned int>("shell_mesh_count") +
    KGShellPolyLoopSurfaceBuilder::Attribute<double>("shell_mesh_power") +
    KGShellPolyLoopSurfaceBuilder::ComplexElement<KGPlanarPolyLoop>("poly_loop");

}  // namespace katrin
