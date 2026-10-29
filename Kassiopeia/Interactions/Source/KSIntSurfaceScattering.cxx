//////////////////////////////////////////////////////////////////////////
// Routine to simulate scattering of an electron on a surface with a given
// probability for backscattering of the electron and a given probability 
// for production of a secondary electron from the surface.
//////////////////////////////////////////////////////////////////////////
#include "KSIntSurfaceScattering.h"

#include "KConst.h"
#include "KRandom.h"
#include "KSInteractionsMessage.h"
#include "KThreeVector.hh"

#include <cmath>
#include <limits>

using katrin::KRandom;
using katrin::KThreeVector;

namespace Kassiopeia
{

namespace
{
std::string GetSurfaceOrSideName(const KSParticle& aParticle)
{
    if (aParticle.GetCurrentSurface() != nullptr) {
        return aParticle.GetCurrentSurface()->GetName();
    }
    if (aParticle.GetCurrentSide() != nullptr) {
        return aParticle.GetCurrentSide()->GetName();
    }
    return "";
}
}  // namespace

KSIntSurfaceScattering::KSIntSurfaceScattering() :
    fScatProbability(std::numeric_limits<double>::quiet_NaN()),
    fScatLossFraction(std::numeric_limits<double>::quiet_NaN()),
    fSecElectronProbability(std::numeric_limits<double>::quiet_NaN()),
    fSecElectronMeanEnergy(std::numeric_limits<double>::quiet_NaN()),
    fSide(KSIntSurfaceScatteringSide::Both)
{}

KSIntSurfaceScattering::KSIntSurfaceScattering(const KSIntSurfaceScattering& aCopy) :
    KSComponent(aCopy),
    fScatProbability(aCopy.fScatProbability),
    fScatLossFraction(aCopy.fScatLossFraction),
    fSecElectronProbability(aCopy.fSecElectronProbability),
    fSecElectronMeanEnergy(aCopy.fSecElectronMeanEnergy),
    fSide(aCopy.fSide)
{}

KSIntSurfaceScattering* KSIntSurfaceScattering::Clone() const
{
    return new KSIntSurfaceScattering(*this);
}

KSIntSurfaceScattering::~KSIntSurfaceScattering() = default;

KThreeVector KSIntSurfaceScattering::GetReflectionDirection(const KSParticle& anInitialParticle, const double tSinTheta, const double tCosTheta, const double tAzimuthalAngle)
{
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
        return KThreeVector(0,0,0);
    }

    const KThreeVector tInitialMomentum = anInitialParticle.GetMomentum();
    KThreeVector tInitialNormalMomentum = tInitialMomentum.Dot(tNormal) * tNormal;
    KThreeVector tInitialTangentMomentum = tInitialMomentum - tInitialNormalMomentum;
    KThreeVector tInitialOrthogonalMomentum = tInitialTangentMomentum.Cross(tInitialNormalMomentum.Unit());

    //define reflected direction
    KThreeVector tReflectedDirection = -1.0 * tCosTheta * tInitialNormalMomentum.Unit();
    tReflectedDirection += tSinTheta * std::cos(tAzimuthalAngle) * tInitialTangentMomentum.Unit();
    tReflectedDirection += tSinTheta * std::sin(tAzimuthalAngle) * tInitialOrthogonalMomentum.Unit();

    return tReflectedDirection;
}

void KSIntSurfaceScattering::ExecuteInteraction(const KSParticle& anInitialParticle,
                                                      KSParticle& aFinalParticle,
                                                      KSParticleQueue& aSecondaries)
{
  double tChoice;

  if (std::isnan(fScatProbability) || std::isnan(fScatLossFraction) || std::isnan(fSecElectronProbability) ||
      std::isnan(fSecElectronMeanEnergy)) {
    intmsg(eError) << "surface diffuse interaction named <" << GetName()
                   << "> is missing a required parameter (ScatProbability, ScatLossFraction, "
                      "SecElectronProbability and SecElectronMeanEnergy must all be set)"
                   << eom;
    return;
  }
  if (!(fScatLossFraction >= 0.0 && fScatLossFraction < 1.0)) {
    intmsg(eError) << "surface diffuse interaction named <" << GetName()
                   << "> is configured for an out-of-range ScatLossFraction of <" << fScatLossFraction << ">."
                   << eom;
    return;
  }

#ifdef Kassiopeia_ENABLE_DEBUG
  intmsg_debug("*************** initial particle" << eom);
  anInitialParticle.Print();

  intmsg_debug("*************** initial particle kinetic energy = " << anInitialParticle.GetKineticEnergy_eV() << " eV" << eom);
#endif
  
  // figure out the basis directions for the particle ejection
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
  if (fSide == KSIntSurfaceScatteringSide::Top && dot_prod > 0) {
    execute_interaction = false;
  }
  if (fSide == KSIntSurfaceScatteringSide::Bottom && dot_prod < 0) {
    execute_interaction = false;
  }

  //only execute interaction if the specified side of this surface is active
  if (!execute_interaction) {
    aFinalParticle = anInitialParticle;
    aFinalParticle.AddLabel(GetName());
    aFinalParticle.AddLabel(GetSurfaceOrSideName(anInitialParticle));
    aFinalParticle.AddLabel("transmitted");
    return;
  }

  tChoice = KRandom::GetInstance().Uniform(0., 1.);
  if (tChoice < fSecElectronProbability) {
    CreateSecondaryElectron(anInitialParticle, aFinalParticle, aSecondaries);
    intmsg_debug("  secondary electron production occurred on child surface <"
                    << GetSurfaceOrSideName(anInitialParticle) << ">" << eom);
  }

  tChoice = KRandom::GetInstance().Uniform(0., 1.);
  if (tChoice < fScatProbability)
  {
    ExecuteReflection(anInitialParticle, aFinalParticle, aSecondaries);
    intmsg_debug("  backscattering occurred on child surface <"
                    << GetSurfaceOrSideName(anInitialParticle) << ">" << eom);
  }
  else
  {
    //kill incoming electron
    aFinalParticle = anInitialParticle;
    aFinalParticle.SetActive(false);
    aFinalParticle.AddLabel(GetName());
    aFinalParticle.AddLabel(GetSurfaceOrSideName(anInitialParticle));
    aFinalParticle.AddLabel("absorbed");
    aFinalParticle.SetMomentum(0., 0., 0.);
    intmsg_debug("  particle absorption occurred on child surface <"
                    << GetSurfaceOrSideName(anInitialParticle) << ">" << eom);
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
  // add new particle to queue  
  //
  auto* tParticle = new KSParticle(anInitialParticle);
  
  //dice direction
  double tAzimuthalAngle = KRandom::GetInstance().Uniform(0., 2. * katrin::KConst::Pi());
  double tSinTheta =
  KRandom::GetInstance().Uniform(0., 0.5);  //this is not a true lambertian (has cut-off angle)
  double tCosTheta = std::sqrt((1.0 - tSinTheta) * (1.0 + tSinTheta));
    
  KThreeVector tDirection = GetReflectionDirection(anInitialParticle, tSinTheta, tCosTheta, tAzimuthalAngle);
  
  // secondary particles particles are emitted predominantly 
  // with very low energies, currently set constant
  double tChildEnergy = fSecElectronMeanEnergy; 
  
  KThreeVector pos = anInitialParticle.GetPosition();
  tParticle->SetPosition(pos);
  
  tParticle->SetMomentum(tDirection);
  tParticle->SetKineticEnergy_eV(tChildEnergy);
  tParticle->SetCurrentSurface(nullptr);
  tParticle->AddLabel(GetName());
  tParticle->AddLabel(GetSurfaceOrSideName(anInitialParticle));
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

    //generate angles for diffuse 'Lambertian' reflection direction
    double tAzimuthalAngle = KRandom::GetInstance().Uniform(0., 2. * katrin::KConst::Pi());
    double tSinTheta = std::sqrt(KRandom::GetInstance().Uniform(
        0.,
        1.));  // only KRandom::GetInstance().Uniform( 0., 1. ); is wrong: see http://www.sciencedirect.com/science/article/pii/S0042207X02001732
    double tCosTheta = std::sqrt((1.0 - tSinTheta) * (1.0 + tSinTheta));
    
    KThreeVector tReflectedMomentum = GetReflectionDirection(anInitialParticle, tSinTheta, tCosTheta, tAzimuthalAngle)
        * anInitialParticle.GetMomentum().Magnitude();

    aFinalParticle = anInitialParticle;
    aFinalParticle.SetMomentum(tReflectedMomentum);
    aFinalParticle.SetKineticEnergy(tKineticEnergy);
    aFinalParticle.AddLabel(GetName());
    aFinalParticle.AddLabel(GetSurfaceOrSideName(anInitialParticle));
    aFinalParticle.AddLabel("backscattered");

    return;
}

}  // namespace Kassiopeia
