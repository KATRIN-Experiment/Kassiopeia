#ifndef Kassiopeia_KSTermMaxLongEnergy_h_
#define Kassiopeia_KSTermMaxLongEnergy_h_

#include "KSTerminator.h"

namespace Kassiopeia
{

class KSParticle;

class KSTermMaxLongEnergy : public KSComponentTemplate<KSTermMaxLongEnergy, KSTerminator>
{
  public:
    KSTermMaxLongEnergy();
    KSTermMaxLongEnergy(const KSTermMaxLongEnergy& aCopy);
    KSTermMaxLongEnergy* Clone() const override;
    ~KSTermMaxLongEnergy() override;

  public:
    void CalculateTermination(const KSParticle& anInitialParticle, bool& aFlag) override;
    void ExecuteTermination(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                            KSParticleQueue& aParticleQueue) const override;

  public:
    void SetMaxLongEnergy(const double& aValue);

  private:
    double fMaxLongEnergy;
};

inline void KSTermMaxLongEnergy::SetMaxLongEnergy(const double& aValue)
{
    fMaxLongEnergy = aValue;
}

}  // namespace Kassiopeia

#endif
