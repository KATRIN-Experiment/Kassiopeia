//////////////////////////////////////////////////////////////////////////
// Routine to simulate scattering of an electron on a surface with a given
// probability for backscattering of the electron and a given probability 
// for production of a secondary electron from the surface.
//////////////////////////////////////////////////////////////////////////
#include "KSIntSurfaceScattering.h"

#include "KRandom.h"
#include "KSInteractionsMessage.h"
using katrin::KRandom;

#include "KConst.h"

#include <cmath>

using katrin::KThreeVector;

namespace Kassiopeia
{

KSIntSurfaceScattering::KSIntSurfaceScattering() :
    fScatProbability(0.3),
    fScatLossFraction(0.0),
    fSecElectronProbability(0.25),
    fSecElectronMeanEnergy(4.0),
    fPerformSideCheck(false),
    fSideSignIsNegative(false),
    fSideName(std::string("both"))
{}

KSIntSurfaceScattering::KSIntSurfaceScattering(const KSIntSurfaceScattering& aCopy) :
    KSComponent(aCopy),
    fScatProbability(aCopy.fScatProbability),
    fScatLossFraction(aCopy.fScatLossFraction),
    fSecElectronProbability(aCopy.fSecElectronProbability),
    fSecElectronMeanEnergy(aCopy.fSecElectronMeanEnergy),
    fPerformSideCheck(aCopy.fPerformSideCheck),
    fSideSignIsNegative(aCopy.fSideSignIsNegative),
    fSideName(aCopy.fSideName)
{}

KSIntSurfaceScattering* KSIntSurfaceScattering::Clone() const
{
    return new KSIntSurfaceScattering(*this);
}

KSIntSurfaceScattering::~KSIntSurfaceScattering() = default;

void KSIntSurfaceScattering::ExecuteInteraction(const KSParticle& anInitialParticle,
                                                      KSParticle& aFinalParticle,
                                                      KSParticleQueue& aSecondaries)
{
  double tChoice;
  KSSurface* tCurrentSurface = anInitialParticle.GetCurrentSurface();

#ifdef Kassiopeia_ENABLE_DEBUG
  intmsg_debug("*************** initial particle" << eom);
  anInitialParticle.Print();

  intmsg_debug("*************** initial particle kinetic energy = " << anInitialParticle.GetKineticEnergy_eV() << " eV" << eom);
#endif
  
  // figure out the basis directions for the particle ejection
  // eject it with a diffuse 'Lambertian' distribution
  KThreeVector tNormal;
  if (anInitialParticle.GetCurrentSurface() != nullptr) {
    tNormal = anInitialParticle.GetCurrentSurface()->Normal(anInitialParticle.GetPosition());
  }
  else if (anInitialParticle.GetCurrentSide() != nullptr) {
    tNormal = anInitialParticle.GetCurrentSide()->Normal(anInitialParticle.GetPosition());
  }
  else {
    intmsg(eError) << "surface diffuse interaction named <" << GetName()
    << "> was given a particle with neither a surface nor a side set" << eom;
    return;
  }
  
  const KThreeVector tInitialMomentum = anInitialParticle.GetMomentum();
  const double dot_prod = tInitialMomentum.Dot(tNormal);
  bool execute_interaction = true;
  if (fPerformSideCheck) {
    if (fSideSignIsNegative && dot_prod > 0) {
      execute_interaction = false;
    }
    if (!fSideSignIsNegative && dot_prod < 0) {
      execute_interaction = false;
    }
  }

  //only execute interaction if the specified side of this surface is active
  if (execute_interaction)
    {
      tChoice = KRandom::GetInstance().Uniform(0., 1.);
      if (tChoice < fSecElectronProbability) {
        CreateSecondaryElectron(anInitialParticle, aFinalParticle, aSecondaries);
        intmsg(eNormal) << "  secondary electron production occurred on child surface <"
                        << (tCurrentSurface != nullptr ? tCurrentSurface->GetName()
                                                       : anInitialParticle.GetCurrentSide()->GetName())
                        << ">" << eom;
      }
    
      tChoice = KRandom::GetInstance().Uniform(0., 1.);
      if (tChoice < fScatProbability)
        {
          ExecuteReflection(anInitialParticle, aFinalParticle, aSecondaries);
          intmsg(eNormal) << "  backscattering occurred on child surface <"
                          << (tCurrentSurface != nullptr ? tCurrentSurface->GetName()
                                                         : anInitialParticle.GetCurrentSide()->GetName())
                          << ">" << eom;
        }
      else
        {
          //kill incoming electron
          aFinalParticle = anInitialParticle;
          aFinalParticle.SetActive(false);
          aFinalParticle.AddLabel("absorbed");
          aFinalParticle.SetMomentum(0., 0., 0.);
          intmsg(eNormal) << "  particle absorption occurred on child surface <"
                          << (tCurrentSurface != nullptr ? tCurrentSurface->GetName()
                                                         : anInitialParticle.GetCurrentSide()->GetName())
                          << ">" << eom;
        }
    }
  else
    {
      aFinalParticle = anInitialParticle;
      aFinalParticle.AddLabel("transmitted");
    }

#ifdef Kassiopeia_ENABLE_DEBUG
  intmsg_debug("*************** final particle" << eom);
  aFinalParticle.Print();
  intmsg_debug("*************** secondaries" << eom);
  for (KSParticle* sec : aSecondaries) sec->Print();
#endif

  return;
}


// Secondary electron production from surface
void KSIntSurfaceScattering::CreateSecondaryElectron(const KSParticle& anInitialParticle,
                                                           KSParticle&,
                                                           KSParticleQueue& aSecondaries)
{
  // figure out the basis directions for the particle ejection
  // eject it with a diffuse 'Lambertian' distribution
  //
  KThreeVector tNormal;
  if (anInitialParticle.GetCurrentSurface() != nullptr) {
    tNormal = anInitialParticle.GetCurrentSurface()->Normal(anInitialParticle.GetPosition());
  }
  else if (anInitialParticle.GetCurrentSide() != nullptr) {
    tNormal = anInitialParticle.GetCurrentSide()->Normal(anInitialParticle.GetPosition());
  }
  else {
    intmsg(eError) << "surface diffuse interaction named <" << GetName()
    << "> was given a particle with neither a surface nor a side set" << eom;
    return;
  }
  
  KThreeVector tInitialMomentum = anInitialParticle.GetMomentum();
  KThreeVector momDirection = tInitialMomentum.Unit();
  
  double dot_prod = tInitialMomentum.Dot(tNormal);
  KThreeVector tInitialNormalMomentum = dot_prod * tNormal;
  
  tInitialNormalMomentum = -1.0 * tInitialNormalMomentum;  //reverse direction for reflection
  KThreeVector tInitialTangentMomentum = tInitialMomentum - tInitialNormalMomentum;
  
  tInitialNormalMomentum = tInitialNormalMomentum.Unit();
  tInitialTangentMomentum = tInitialTangentMomentum.Unit();
  KThreeVector tInitialOrthogonalMomentum = tInitialTangentMomentum.Cross(tInitialNormalMomentum.Unit());

  // add new particle to queue  
  //
  auto* tParticle = new KSParticle(anInitialParticle);
  
  //dice direction
  double tAzimuthalAngle = KRandom::GetInstance().Uniform(0., 2. * katrin::KConst::Pi());
  double tSinTheta =
  KRandom::GetInstance().Uniform(0., 0.5);  //this is not a true lambertian (has cut-off angle)
  double tCosTheta = std::sqrt((1.0 - tSinTheta) * (1.0 + tSinTheta));
  
  KThreeVector tDirection;
  tDirection = tCosTheta * tInitialNormalMomentum;
  tDirection += tSinTheta * std::cos(tAzimuthalAngle) * tInitialTangentMomentum.Unit();
  tDirection += tSinTheta * std::sin(tAzimuthalAngle) * tInitialOrthogonalMomentum.Unit();
  
  if (tDirection.Dot(momDirection) > 0) {
  tDirection = -1.0 * tDirection;
  }
  
  // secondary particles particles are emitted predominantly 
  // with very low energies, currently set constant
  double tChildEnergy = fSecElectronMeanEnergy; 
  
  KThreeVector pos = anInitialParticle.GetPosition();
  tParticle->SetPosition(pos);
  
  tParticle->SetMomentum(tDirection);
  tParticle->SetKineticEnergy_eV(tChildEnergy);
  tParticle->SetCurrentSurface(nullptr);
  tParticle->AddLabel("secondary");
  aSecondaries.push_back(tParticle);
  
  return;
}


// Electron backscattering from surface, adapted from diffuse reflection
void KSIntSurfaceScattering::ExecuteReflection(const KSParticle& anInitialParticle,
                                                     KSParticle& aFinalParticle,
                                                     KSParticleQueue&)
{
    double tKineticEnergy = anInitialParticle.GetKineticEnergy();

    // energy loss due to backscattering
    tKineticEnergy *= (1.0 - fScatLossFraction);

    //prevent kinetic energy from going negative
    if (tKineticEnergy < 0.0) {
        intmsg(eError) << "surface diffuse interaction named <" << GetName()
                       << "> tried to give a particle a negative kinetic energy." << eom;
        return;
    }

    //generate angles for diffuse 'Lambertian' reflection direction
    double tAzimuthalAngle = KRandom::GetInstance().Uniform(0., 2. * katrin::KConst::Pi());
    double tSinTheta = std::sqrt(KRandom::GetInstance().Uniform(
        0.,
        1.));  // only KRandom::GetInstance().Uniform( 0., 1. ); is wrong: see http://www.sciencedirect.com/science/article/pii/S0042207X02001732
    double tCosTheta = std::sqrt((1.0 - tSinTheta) * (1.0 + tSinTheta));

    KThreeVector tNormal;
    if (anInitialParticle.GetCurrentSurface() != nullptr) {
        tNormal = anInitialParticle.GetCurrentSurface()->Normal(anInitialParticle.GetPosition());
    }
    else if (anInitialParticle.GetCurrentSide() != nullptr) {
        tNormal = anInitialParticle.GetCurrentSide()->Normal(anInitialParticle.GetPosition());
    }
    else {
        intmsg(eError) << "surface diffuse interaction named <" << GetName()
                       << "> was given a particle with neither a surface nor a side set" << eom;
        return;
    }

    KThreeVector tInitialMomentum = anInitialParticle.GetMomentum();
    double tMomentumMagnitude = tInitialMomentum.Magnitude();

    KThreeVector tInitialNormalMomentum = tInitialMomentum.Dot(tNormal) * tNormal;
    KThreeVector tInitialTangentMomentum = tInitialMomentum - tInitialNormalMomentum;
    KThreeVector tInitialOrthogonalMomentum = tInitialTangentMomentum.Cross(tInitialNormalMomentum.Unit());

    //define reflected direction
    KThreeVector tReflectedDirection = -1.0 * tCosTheta * tInitialNormalMomentum.Unit();
    tReflectedDirection += tSinTheta * std::cos(tAzimuthalAngle) * tInitialTangentMomentum.Unit();
    tReflectedDirection += tSinTheta * std::sin(tAzimuthalAngle) * tInitialOrthogonalMomentum.Unit();

    KThreeVector tReflectedMomentum = tMomentumMagnitude * tReflectedDirection;

    aFinalParticle = anInitialParticle;
    aFinalParticle.SetMomentum(tReflectedMomentum);
    aFinalParticle.SetKineticEnergy(tKineticEnergy);
    aFinalParticle.AddLabel("backscattered");

    return;
}

}  // namespace Kassiopeia
