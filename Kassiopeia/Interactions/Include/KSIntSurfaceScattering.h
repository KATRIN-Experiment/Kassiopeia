//////////////////////////////////////////////////////////////////////////
// Routine to simulate scattering of an electron on a surface with a given
// probability for backscattering of the electron and a given probability 
// for production of a secondary electron from the surface.
//////////////////////////////////////////////////////////////////////////
#ifndef Kassiopeia_KSIntSurfaceScattering_h_
#define Kassiopeia_KSIntSurfaceScattering_h_

#include "KField.h"
#include "KSInteractionsMessage.h"
#include "KSSurfaceInteraction.h"
#include "KThreeVector.hh"

#include <string>
namespace Kassiopeia
{

class KSStep;

enum class KSIntSurfaceScatteringSide
{
    Both,
    Top,
    Bottom
};

class KSIntSurfaceScattering : public KSComponentTemplate<KSIntSurfaceScattering, KSSurfaceInteraction>
{
  public:
    KSIntSurfaceScattering();
    KSIntSurfaceScattering(const KSIntSurfaceScattering& aCopy);
    KSIntSurfaceScattering* Clone() const override;
    ~KSIntSurfaceScattering() override;

  public:
    void ExecuteInteraction(const KSParticle& anInitialParticle,
                                  KSParticle& aFinalParticle,
                                  KSParticleQueue& aSecondaries) override;
    void CreateSecondaryElectron(const KSParticle& anInitialParticle,
                                       KSParticle& aFinalParticle,
                                       KSParticleQueue& aSecondaries);
    void ExecuteReflection(const KSParticle& anInitialParticle,
                                 KSParticle& aFinalParticle,
                                 KSParticleQueue& aSecondaries);
    katrin::KThreeVector GetReflectionDirection(const KSParticle& anInitialParticle,
                                        const double tSinTheta,
                                        const double tCosTheta,
                                        const double tAzimuthalAngle);

    void SetSide(std::string side_name)
    {
        if (side_name == "both") {
            fSide = KSIntSurfaceScatteringSide::Both;
            return;
        }
        
        if (side_name == "top") {
            fSide = KSIntSurfaceScatteringSide::Top;
            return;
        }
        
        if (side_name == "bottom") {
            fSide = KSIntSurfaceScatteringSide::Bottom;
            return;
        }
        
       intmsg(eError) << "surface diffuse interaction named <" << GetName()
        << "> does not understand side <" << side_name
        << ">. Valid sides are: both, top, bottom." << eom;
       return;
    }

  public:
    K_SET_GET(double, ScatProbability)
    K_SET_GET(double, ScatLossFraction)
    K_SET_GET(double, SecElectronProbability)
    K_SET_GET(double, SecElectronMeanEnergy)
    K_SET_GET(KSIntSurfaceScatteringSide, Side)
};

}  // namespace Kassiopeia

#endif
