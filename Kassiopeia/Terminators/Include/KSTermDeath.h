#ifndef Kassiopeia_KSTermDeath_h_
#define Kassiopeia_KSTermDeath_h_

#include "KSTerminator.h"

namespace Kassiopeia
{

class KSTermDeath : public KSComponentTemplate<KSTermDeath, KSTerminator>
{
  public:
    KSTermDeath();
    KSTermDeath(const KSTermDeath& aCopy);
    KSTermDeath* Clone() const override;
    ~KSTermDeath() override;

  public:
    void CalculateTermination(const KSParticle& anInitialParticle, bool& aFlag) override;
    void ExecuteTermination(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                            KSParticleQueue& aParticleQueue) const override;
};

}  // namespace Kassiopeia

#endif
