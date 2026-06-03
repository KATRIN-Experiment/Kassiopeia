//////////////////////////////////////////////////////////////////////////
// Routine to simulate scattering of an electron on a surface with a given
// probability for backscattering of the electron and a given probability 
// for production of a secondary electron from the surface.
//////////////////////////////////////////////////////////////////////////
#ifndef Kassiopeia_KSIntSurfaceScattering_h_
#define Kassiopeia_KSIntSurfaceScattering_h_

#include "KField.h"
#include "KSSurfaceInteraction.h"

namespace Kassiopeia
{

class KSStep;

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

    void SetSide(std::string side_name)
    {
        //default is both sides of the surface execute the interaction
        fSideName = std::string("both");
        fPerformSideCheck = false;

        //top is the side on which the normal points outward
        if (side_name == "top") {
            fSideName = side_name;
            fPerformSideCheck = true;
            fSideSignIsNegative = true;
        }

        //bottom is the side on which the normal std::vector points inward
        if (side_name == "bottom") {
            fSideName = side_name;
            fPerformSideCheck = true;
            fSideSignIsNegative = false;
        }
    }

    std::string GetSide() const
    {
        return fSideName;
    };

  public:
    K_SET_GET(double, ScatProbability)
    K_SET_GET(double, ScatLossFraction)
    K_SET_GET(double, SecElectronProbability)
    K_SET_GET(double, SecElectronMeanEnergy)

  private:
    bool fPerformSideCheck;
    bool fSideSignIsNegative;
    std::string fSideName;
};

}  // namespace Kassiopeia

#endif
