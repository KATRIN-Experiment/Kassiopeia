#ifndef Kassiopeia_KSTermMaxTime_h_
#define Kassiopeia_KSTermMaxTime_h_

#include "KSTerminator.h"

namespace Kassiopeia
{

class KSParticle;

class KSTermMaxTime : public KSComponentTemplate<KSTermMaxTime, KSTerminator>
{
  public:
    KSTermMaxTime();
    KSTermMaxTime(const KSTermMaxTime& aCopy);
    KSTermMaxTime* Clone() const override;
    ~KSTermMaxTime() override;

  public:
    void CalculateTermination(const KSParticle& anInitialParticle, bool& aFlag) override;
    void ExecuteTermination(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                            KSParticleQueue& aParticleQueue) const override;

  public:
    void SetTime(const double& aValue);

  private:
    double fTime;
};

inline void KSTermMaxTime::SetTime(const double& aValue)
{
    fTime = aValue;
}

}  // namespace Kassiopeia

#endif
