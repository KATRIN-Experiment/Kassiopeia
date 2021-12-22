#ifndef Kassiopeia_KSIntSpinFlip_h_
#define Kassiopeia_KSIntSpinFlip_h_

#include "KSSpaceInteraction.h"

#include <vector>

namespace Kassiopeia
{

class KSIntSpinFlipCalculator;

class KSIntSpinFlip : public KSComponentTemplate<KSIntSpinFlip, KSSpaceInteraction>
{
  public:
    KSIntSpinFlip();
    KSIntSpinFlip(const KSIntSpinFlip& aCopy);
    KSIntSpinFlip* Clone() const override;
    ~KSIntSpinFlip() override;

  public:
    void CalculateTransitionRate(const KSParticle& aParticle, double& aTransitionRate);

    void CalculateInteraction(const KSTrajectory& aTrajectory, const KSParticle& aTrajectoryInitialParticle,
                              const KSParticle& aTrajectoryFinalParticle,
                              const katrin::KThreeVector& aTrajectoryCenter, const double& aTrajectoryRadius,
                              const double& aTrajectoryTimeStep, KSParticle& anInteractionParticle, double& aTimeStep,
                              bool& aFlag) override;

    void ExecuteInteraction(const KSParticle& anInteractionParticle, KSParticle& aFinalParticle,
                            KSParticleQueue& aSecondaries) const override;
};

}  // namespace Kassiopeia

#endif
