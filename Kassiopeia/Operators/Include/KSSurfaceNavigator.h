#ifndef Kassiopeia_KSSurfaceNavigator_h_
#define Kassiopeia_KSSurfaceNavigator_h_

#include "KSComponentTemplate.h"
#include "KSParticle.h"

namespace Kassiopeia
{

class KSSurfaceNavigator : public KSComponentTemplate<KSSurfaceNavigator>
{
  public:
    KSSurfaceNavigator();
    ~KSSurfaceNavigator() override;

  public:
    virtual void ExecuteNavigation(const KSParticle& anInitialParticle, const KSParticle& aNavigationParticle,
                                   KSParticle& aFinalParticle, KSParticleQueue& aSecondaries) const = 0;

    virtual void FinalizeNavigation(KSParticle& aFinalParticle) const = 0;
};

}  // namespace Kassiopeia

#endif
