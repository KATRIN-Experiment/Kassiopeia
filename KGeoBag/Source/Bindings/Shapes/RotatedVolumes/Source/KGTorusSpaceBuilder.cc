#include "KGTorusSpaceBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

STATICINT sKGTorusSpaceBuilder = KGInterfaceBuilder::ComplexElement<KGTorusSpace>("torus_space");

STATICINT sKGTorusSpaceBuilderStructure = KGTorusSpaceBuilder::Attribute<std::string>("name") +
                                          //! z coordinate of the center in meters
                                          KGTorusSpaceBuilder::Attribute<double>("z") +
                                          //! r coordinate of the center in meters
                                          KGTorusSpaceBuilder::Attribute<double>("r") +
                                          //! the toroidal radius in meters
                                          KGTorusSpaceBuilder::Attribute<double>("radius") +
                                          //! toroidal mesh parameter (default is 64)
                                          KGTorusSpaceBuilder::Attribute<unsigned int>("toroidal_mesh_count") +
                                          //! axial mesh parameter (default is 64)
                                          KGTorusSpaceBuilder::Attribute<unsigned int>("axial_mesh_count");

}  // namespace katrin
