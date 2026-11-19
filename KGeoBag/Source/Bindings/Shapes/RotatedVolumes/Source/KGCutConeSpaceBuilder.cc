#include "KGCutConeSpaceBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

STATICINT sKGCutConeSpaceBuilder = KGInterfaceBuilder::ComplexElement<KGCutConeSpace>("cut_cone_space");

STATICINT sKGCutConeSpaceBuilderStructure = KGCutConeSpaceBuilder::Attribute<std::string>("name") +
                                            //! the first  z coordinate in meters
                                            KGCutConeSpaceBuilder::Attribute<double>("z1") +
                                            //! the second z coordinate in meters
                                            KGCutConeSpaceBuilder::Attribute<double>("z2") +
                                            //! the first r coordinate in meters
                                            KGCutConeSpaceBuilder::Attribute<double>("r1") +
                                            //! the second r coordinate in meters
                                            KGCutConeSpaceBuilder::Attribute<double>("r2") +
                                            //! longitudinal mesh parameter (default is 1)
                                            KGCutConeSpaceBuilder::Attribute<unsigned int>("longitudinal_mesh_count") +
                                            //! longitudinal meshing power (default is 1.)
                                            KGCutConeSpaceBuilder::Attribute<double>("longitudinal_mesh_power") +
                                            //! radial mesh parameter (default is 1)
                                            KGCutConeSpaceBuilder::Attribute<unsigned int>("radial_mesh_count") +
                                            //! radial meshing power (default is 1.)
                                            KGCutConeSpaceBuilder::Attribute<double>("radial_mesh_power") +
                                            //! axial mesh parameter (default is 16)
                                            KGCutConeSpaceBuilder::Attribute<unsigned int>("axial_mesh_count");
}  // namespace katrin
