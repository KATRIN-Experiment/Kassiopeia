#include "KGBoxSpaceBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

STATICINT sKGBoxSpaceBuilder = KGInterfaceBuilder::ComplexElement<KGBoxSpace>("box_space");

STATICINT sKGBoxSpaceBuilderStructure = KGBoxSpaceBuilder::Attribute<std::string>("name") +
                                        //! one of the x extrema in meters
                                        KGBoxSpaceBuilder::Attribute<double>("xa") +
                                        //! the other x extremum in meters
                                        KGBoxSpaceBuilder::Attribute<double>("xb") +
                                        //! x mesh count
                                        KGBoxSpaceBuilder::Attribute<unsigned int>("x_mesh_count") +
                                        //! x mesh power
                                        KGBoxSpaceBuilder::Attribute<double>("x_mesh_power") +
                                        //! one of the y extrema in meters
                                        KGBoxSpaceBuilder::Attribute<double>("ya") +
                                        //! the other y extremum in meters
                                        KGBoxSpaceBuilder::Attribute<double>("yb") +
                                        //! y mesh count
                                        KGBoxSpaceBuilder::Attribute<unsigned int>("y_mesh_count") +
                                        //! y mesh power
                                        KGBoxSpaceBuilder::Attribute<double>("y_mesh_power") +
                                        //! one of the z extrema in meters
                                        KGBoxSpaceBuilder::Attribute<double>("za") +
                                        //! the other z extremum in meters
                                        KGBoxSpaceBuilder::Attribute<double>("zb") +
                                        //! z mesh count
                                        KGBoxSpaceBuilder::Attribute<unsigned int>("z_mesh_count") +
                                        //! z mesh power
                                        KGBoxSpaceBuilder::Attribute<double>("z_mesh_power");

}  // namespace katrin
