//////////////////////////////////////////////////////////////////////////
// Routine to simulate scattering of an electron on a surface with a given
// probability for backscattering of the electron and a given probability 
// for production of a secondary electron from the surface.
//////////////////////////////////////////////////////////////////////////
#ifndef Kassiopeia_KSIntSurfaceScatteringBuilder_h_
#define Kassiopeia_KSIntSurfaceScatteringBuilder_h_

#include "KComplexElement.hh"
#include "KSIntSurfaceScattering.h"

using namespace Kassiopeia;
namespace katrin
{
  typedef KComplexElement<KSIntSurfaceScattering> KSIntSurfaceScatteringBuilder;
  
  template<> inline bool KSIntSurfaceScatteringBuilder::AddAttribute(KContainer* aContainer)
    {
      if (aContainer->GetName() == "name") {
          aContainer->CopyTo(fObject, &KNamed::SetName);
          return true;
      }
      if (aContainer->GetName() == "sec_electron_probability") {
          aContainer->CopyTo(fObject, &KSIntSurfaceScattering::SetSecElectronProbability);
          return true;
      }
      if (aContainer->GetName() == "sec_electron_mean_energy") {
          aContainer->CopyTo(fObject, &KSIntSurfaceScattering::SetSecElectronMeanEnergy);
          return true;
      }
      if (aContainer->GetName() == "scat_probability") {
          aContainer->CopyTo(fObject, &KSIntSurfaceScattering::SetScatProbability);
          return true;
      }
      if (aContainer->GetName() == "scat_loss_fraction") {
          aContainer->CopyTo(fObject, &KSIntSurfaceScattering::SetScatLossFraction);
          return true;
      }
      if (aContainer->GetName() == "side") {
        aContainer->CopyTo(fObject, &KSIntSurfaceScattering::SetSide);
        return true;
      }
      return false;
    }
}  // namespace katrin

#endif
