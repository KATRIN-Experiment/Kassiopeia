#ifndef Kassiopeia_KSTermMaxZ_h_
#define Kassiopeia_KSTermMaxZ_h_

#include "KSTerminator.h"

namespace Kassiopeia
{

class KSParticle;

class KSTermMaxZ : public KSComponentTemplate<KSTermMaxZ, KSTerminator>
{
  public:
    KSTermMaxZ();
    KSTermMaxZ(const KSTermMaxZ& aCopy);
    KSTermMaxZ* Clone() const override;
    ~KSTermMaxZ() override;

  public:
    void CalculateTermination(const KSParticle& anInitialParticle, bool& aFlag) override;
    void ExecuteTermination(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                            KSParticleQueue& aParticleQueue) const override;

  public:
    void SetMaxZ(const double& aValue);

  private:
    double fMaxZ;
};

inline void KSTermMaxZ::SetMaxZ(const double& aValue)
{
    fMaxZ = aValue;
}

}  // namespace Kassiopeia

#endif
