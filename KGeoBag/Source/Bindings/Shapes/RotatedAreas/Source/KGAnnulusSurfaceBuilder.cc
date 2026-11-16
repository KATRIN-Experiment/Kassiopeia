#include "KGAnnulusSurfaceBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

STATICINT sKGAnnulusSurfaceBuilder = KGInterfaceBuilder::ComplexElement<KGAnnulusSurface>("annulus_surface");

STATICINT sKGAnnulusSurfaceBuilderStructure = KGAnnulusSurfaceBuilder::Attribute<std::string>("name") +
                                              //! z-position in meters
                                              KGAnnulusSurfaceBuilder::Attribute<double>("z") +
                                              //! the first of the radii in meters
                                              KGAnnulusSurfaceBuilder::Attribute<double>("r1") +
                                              //! the second radius in meters
                                              KGAnnulusSurfaceBuilder::Attribute<double>("r2") +
                                              //! radial mesh parameter (default is 1)
                                              KGAnnulusSurfaceBuilder::Attribute<unsigned int>("radial_mesh_count") +
                                              //! radial meshing power (default is 1.)
                                              KGAnnulusSurfaceBuilder::Attribute<double>("radial_mesh_power") +
                                              //! axial mesh parameter (default is 16)
                                              KGAnnulusSurfaceBuilder::Attribute<unsigned int>("axial_mesh_count");

}  // namespace katrin
