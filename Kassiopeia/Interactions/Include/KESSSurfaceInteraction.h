#ifndef Kassiopeia_KESSSurfaceInteraction_h_
#define Kassiopeia_KESSSurfaceInteraction_h_

#include "KField.h"
#include "KSParticle.h"
#include "KSSurfaceInteraction.h"

namespace Kassiopeia
{

class KESSSurfaceInteraction : public KSComponentTemplate<KESSSurfaceInteraction, KSSurfaceInteraction>
{
  public:
    KESSSurfaceInteraction();
    KESSSurfaceInteraction(const KESSSurfaceInteraction& aCopy);
    KESSSurfaceInteraction* Clone() const override;
    ~KESSSurfaceInteraction() override;


    void ExecuteInteraction(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                            KSParticleQueue& aQueue) override;


    void ExecuteTransmission(const KSParticle& anInitialParticle, KSParticle& aFinalParticle);

    void ExecuteReflection(const KSParticle& anInitialParticle, KSParticle& aFinalParticle);

    double CalculateTransmissionProbability(const double aKineticEnergy, const double aCosIncidentAngle);

    typedef enum  // NOLINT(modernize-use-using)
    {
        eEnteringSilicon,
        eExitingSilicon
    } ElectronDirection;
    ElectronDirection fElectronDirection;

    typedef enum  // NOLINT(modernize-use-using)
    {
        eNormalPointingSilicon,
        eNormalPointingAway
    } SurfaceOrientation;

    K_SET_GET(double, ElectronAffinity)

    K_SET_GET(SurfaceOrientation, SurfaceOrientation)
};
}  // namespace Kassiopeia

#endif  //Kassiopeia_KESSSurfaceInteraction_h_
