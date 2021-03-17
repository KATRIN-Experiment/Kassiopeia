#include "KGRotatedPolyLoopSpaceBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

STATICINT sKGRotatedPolyLoopSpaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGRotatedPolyLoopSpace>("rotated_poly_loop_space");

STATICINT sKGRotatedPolyLoopSpaceBuilderStructure =
    KGRotatedPolyLoopSpaceBuilder::Attribute<std::string>("name") +
    KGRotatedPolyLoopSpaceBuilder::Attribute<unsigned int>("rotated_mesh_count") +
    KGRotatedPolyLoopSpaceBuilder::ComplexElement<KGPlanarPolyLoop>("poly_loop");

}  // namespace katrin
