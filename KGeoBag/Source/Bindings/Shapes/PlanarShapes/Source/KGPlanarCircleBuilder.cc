#include "KGPlanarCircleBuilder.hh"

namespace katrin
{

STATICINT sKGPlanarCircleBuilderStructure = KGPlanarCircleBuilder::Attribute<double>("x") +
                                            KGPlanarCircleBuilder::Attribute<double>("y") +
                                            KGPlanarCircleBuilder::Attribute<double>("radius") +
                                            KGPlanarCircleBuilder::Attribute<unsigned int>("circle_mesh_count");

}
