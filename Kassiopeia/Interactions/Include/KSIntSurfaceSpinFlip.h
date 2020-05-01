#ifndef Kassiopeia_KSIntSurfaceSpinFlip_h_
#define Kassiopeia_KSIntSurfaceSpinFlip_h_

#include "KField.h"
#include "KSSurfaceInteraction.h"

namespace Kassiopeia
{

class KSStep;

class KSIntSurfaceSpinFlip : public KSComponentTemplate<KSIntSurfaceSpinFlip, KSSurfaceInteraction>
{
  public:
    KSIntSurfaceSpinFlip();
    KSIntSurfaceSpinFlip(const KSIntSurfaceSpinFlip& aCopy);
    KSIntSurfaceSpinFlip* Clone() const override;
    ~KSIntSurfaceSpinFlip() override;

  public:
    void ExecuteInteraction(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                            KSParticleQueue& aSecondaries) override;

  public:
    K_SET_GET(double, Probability)
};

}  // namespace Kassiopeia

#endif
