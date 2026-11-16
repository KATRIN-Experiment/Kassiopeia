#include "KGDiskSurfaceBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

STATICINT sKGDiskSurfaceBuilder = KGInterfaceBuilder::ComplexElement<KGDiskSurface>("disk_surface");

STATICINT sKGDiskSurfaceBuilderStructure = KGDiskSurfaceBuilder::Attribute<std::string>("name") +
                                           //! radius in meters
                                           KGDiskSurfaceBuilder::Attribute<double>("r") +
                                           //! z-position in meters
                                           KGDiskSurfaceBuilder::Attribute<double>("z") +
                                           //! radial mesh parameter (default is 1)
                                           KGDiskSurfaceBuilder::Attribute<unsigned int>("radial_mesh_count") +
                                           //! radial meshing power (default is 1.)
                                           KGDiskSurfaceBuilder::Attribute<double>("radial_mesh_power") +
                                           //! axial mesh parameter (default is 16)
                                           KGDiskSurfaceBuilder::Attribute<unsigned int>("axial_mesh_count");

}  // namespace katrin
