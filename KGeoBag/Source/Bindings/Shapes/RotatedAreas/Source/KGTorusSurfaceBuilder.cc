#include "KGTorusSurfaceBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

STATICINT sKGTorusSurfaceBuilder = KGInterfaceBuilder::ComplexElement<KGTorusSurface>("torus_surface");

STATICINT sKGTorusSurfaceBuilderStructure = KGTorusSurfaceBuilder::Attribute<std::string>("name") +
                                            //! z coordinate of the center in meters
                                            KGTorusSurfaceBuilder::Attribute<double>("z") +
                                            //! r coordinate of the center in meters
                                            KGTorusSurfaceBuilder::Attribute<double>("r") +
                                            //! the toroidal radius in meters
                                            KGTorusSurfaceBuilder::Attribute<double>("radius") +
                                            //! toroidal mesh parameter (default is 64)
                                            KGTorusSurfaceBuilder::Attribute<unsigned int>("toroidal_mesh_count") +
                                            //! axial mesh parameter (default is 64)
                                            KGTorusSurfaceBuilder::Attribute<unsigned int>("axial_mesh_count");

}  // namespace katrin
