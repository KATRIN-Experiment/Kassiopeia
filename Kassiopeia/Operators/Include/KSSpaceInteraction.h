#ifndef Kassiopeia_KSSpaceInteraction_h_
#define Kassiopeia_KSSpaceInteraction_h_

#include "KSComponentTemplate.h"
#include "KSParticle.h"
#include "KSTrajectory.h"

namespace Kassiopeia
{

class KSSpaceInteraction : public KSComponentTemplate<KSSpaceInteraction>
{
  public:
    KSSpaceInteraction();
    ~KSSpaceInteraction() override;

  public:
    virtual void CalculateInteraction(const KSTrajectory& aTrajectory, const KSParticle& aTrajectoryInitialParticle,
                                      const KSParticle& aTrajectoryFinalParticle,
                                      const KGeoBag::KThreeVector& aTrajectoryCenter, const double& aTrajectoryRadius,
                                      const double& aTrajectoryTimeStep, KSParticle& anInteractionParticle,
                                      double& anInteractionStep, bool& anInteractionFlag) = 0;

    virtual void ExecuteInteraction(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                                    KSParticleQueue& aSecondaries) const = 0;
};

}  // namespace Kassiopeia

#endif
