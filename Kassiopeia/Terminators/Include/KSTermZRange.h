#ifndef Kassiopeia_KSTermZRange_h_
#define Kassiopeia_KSTermZRange_h_

#include "KSTerminator.h"

namespace Kassiopeia
{

class KSParticle;

class KSTermZRange : public KSComponentTemplate<KSTermZRange, KSTerminator>
{
  public:
    KSTermZRange();
    KSTermZRange(const KSTermZRange& aCopy);
    KSTermZRange* Clone() const override;
    ~KSTermZRange() override;

  public:
    void CalculateTermination(const KSParticle& anInitialParticle, bool& aFlag) override;
    void ExecuteTermination(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                            KSParticleQueue& aParticleQueue) const override;

  public:
    void SetMaxZ(const double& aValue);
    void SetMinZ(const double& aValue);

  private:
    double fMinZ;
    double fMaxZ;
};

inline void KSTermZRange::SetMinZ(const double& aValue)
{
    fMinZ = aValue;
}
inline void KSTermZRange::SetMaxZ(const double& aValue)
{
    fMaxZ = aValue;
}

}  // namespace Kassiopeia

#endif
