//////////////////////////////////////////////////////////////////////////
// Routine to simulate scattering of an electron on a surface with a given
// probability for backscattering of the electron and a given probability 
// for production of a secondary electron from the surface.
//////////////////////////////////////////////////////////////////////////
#include "KSIntSurfaceScatteringBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{
  template<> KSIntSurfaceScatteringBuilder::~KComplexElement() = default;
  
  STATICINT sKSIntSurfaceScatteringStructure = KSIntSurfaceScatteringBuilder::Attribute<std::string>("name") +
                                               KSIntSurfaceScatteringBuilder::Attribute<double>("sec_electron_probability") +
                                               KSIntSurfaceScatteringBuilder::Attribute<double>("sec_electron_mean_energy") +
                                               KSIntSurfaceScatteringBuilder::Attribute<double>("scat_probability") +
                                               KSIntSurfaceScatteringBuilder::Attribute<double>("scat_loss_fraction") +
                                               KSIntSurfaceScatteringBuilder::Attribute<std::string>("side");
  
  STATICINT sKSIntSurfaceScatteringElement = KSRootBuilder::ComplexElement<KSIntSurfaceScattering>("ksint_surface_scattering");
}  // namespace katrin
