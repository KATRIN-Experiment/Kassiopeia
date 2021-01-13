#include "KSIntSurfaceDiffuseBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSIntSurfaceDiffuseBuilder::~KComplexElement() = default;

STATICINT sKSIntSurfaceDiffuseStructure = KSIntSurfaceDiffuseBuilder::Attribute<std::string>("name") +
                                          KSIntSurfaceDiffuseBuilder::Attribute<double>("reflection_loss") +
                                          KSIntSurfaceDiffuseBuilder::Attribute<double>("transmission_loss") +
                                          KSIntSurfaceDiffuseBuilder::Attribute<double>("reflection_loss_fraction") +
                                          KSIntSurfaceDiffuseBuilder::Attribute<double>("transmission_loss_fraction") +
                                          KSIntSurfaceDiffuseBuilder::Attribute<double>("probability");

STATICINT sKSIntSurfaceDiffuseElement = KSRootBuilder::ComplexElement<KSIntSurfaceDiffuse>("ksint_surface_diffuse");
}  // namespace katrin
