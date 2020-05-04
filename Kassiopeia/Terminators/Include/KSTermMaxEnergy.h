#ifndef Kassiopeia_KSTermMaxEnergy_h_
#define Kassiopeia_KSTermMaxEnergy_h_

#include "KSTerminator.h"

namespace Kassiopeia
{

class KSParticle;

class KSTermMaxEnergy : public KSComponentTemplate<KSTermMaxEnergy, KSTerminator>
{
  public:
    KSTermMaxEnergy();
    KSTermMaxEnergy(const KSTermMaxEnergy& aCopy);
    KSTermMaxEnergy* Clone() const override;
    ~KSTermMaxEnergy() override;

  public:
    void CalculateTermination(const KSParticle& anInitialParticle, bool& aFlag) override;
    void ExecuteTermination(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                            KSParticleQueue& aParticleQueue) const override;

  public:
    void SetMaxEnergy(const double& aValue);

  private:
    double fMaxEnergy;
};

inline void KSTermMaxEnergy::SetMaxEnergy(const double& aValue)
{
    fMaxEnergy = aValue;
}

}  // namespace Kassiopeia

#endif
