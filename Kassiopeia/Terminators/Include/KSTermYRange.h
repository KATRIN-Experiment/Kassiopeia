#ifndef Kassiopeia_KSTermYRange_h_
#define Kassiopeia_KSTermYRange_h_

#include "KSTerminator.h"

namespace Kassiopeia
{

class KSParticle;

class KSTermYRange : public KSComponentTemplate<KSTermYRange, KSTerminator>
{
  public:
    KSTermYRange();
    KSTermYRange(const KSTermYRange& aCopy);
    KSTermYRange* Clone() const override;
    ~KSTermYRange() override;

  public:
    void CalculateTermination(const KSParticle& anInitialParticle, bool& aFlag) override;
    void ExecuteTermination(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                            KSParticleQueue& aParticleQueue) const override;

  public:
    void SetMaxY(const double& aValue);
    void SetMinY(const double& aValue);

  private:
    double fMinY;
    double fMaxY;
};

inline void KSTermYRange::SetMinY(const double& aValue)
{
    fMinY = aValue;
}
inline void KSTermYRange::SetMaxY(const double& aValue)
{
    fMaxY = aValue;
}

}  // namespace Kassiopeia

#endif
