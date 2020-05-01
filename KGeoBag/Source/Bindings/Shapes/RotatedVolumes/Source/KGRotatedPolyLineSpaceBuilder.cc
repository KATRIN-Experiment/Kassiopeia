#include "KGRotatedPolyLineSpaceBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

STATICINT sKGRotatedPolyLineSpaceBuilder =
    KGInterfaceBuilder::ComplexElement<KGRotatedPolyLineSpace>("rotated_poly_line_space");

STATICINT sKGRotatedPolyLineSpaceBuilderStructure =
    KGRotatedPolyLineSpaceBuilder::Attribute<string>("name") +
    KGRotatedPolyLineSpaceBuilder::Attribute<unsigned int>("rotated_mesh_count") +
    KGRotatedPolyLineSpaceBuilder::Attribute<unsigned int>("flattened_mesh_count") +
    KGRotatedPolyLineSpaceBuilder::Attribute<double>("flattened_mesh_power") +
    KGRotatedPolyLineSpaceBuilder::ComplexElement<KGPlanarPolyLine>("poly_line");

}  // namespace katrin
