#include "KGDiskSurfaceBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

STATICINT sKGDiskSurfaceBuilder = KGInterfaceBuilder::ComplexElement<KGDiskSurface>("disk_surface");

STATICINT sKGDiskSurfaceBuilderStructure =
    KGDiskSurfaceBuilder::Attribute<std::string>("name") + KGDiskSurfaceBuilder::Attribute<double>("r") +
    KGDiskSurfaceBuilder::Attribute<double>("z") + KGDiskSurfaceBuilder::Attribute<unsigned int>("radial_mesh_count") +
    KGDiskSurfaceBuilder::Attribute<double>("radial_mesh_power") +
    KGDiskSurfaceBuilder::Attribute<unsigned int>("axial_mesh_count");

}  // namespace katrin
