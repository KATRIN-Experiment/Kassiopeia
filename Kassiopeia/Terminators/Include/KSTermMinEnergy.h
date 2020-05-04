#ifndef Kassiopeia_KSTermMinEnergy_h_
#define Kassiopeia_KSTermMinEnergy_h_

#include "KSTerminator.h"

namespace Kassiopeia
{

class KSParticle;

class KSTermMinEnergy : public KSComponentTemplate<KSTermMinEnergy, KSTerminator>
{
  public:
    KSTermMinEnergy();
    KSTermMinEnergy(const KSTermMinEnergy& aCopy);
    KSTermMinEnergy* Clone() const override;
    ~KSTermMinEnergy() override;

  public:
    void CalculateTermination(const KSParticle& anInitialParticle, bool& aFlag) override;
    void ExecuteTermination(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                            KSParticleQueue& aParticleQueue) const override;

  public:
    void SetMinEnergy(const double& aValue);

  private:
    double fMinEnergy;
};

inline void KSTermMinEnergy::SetMinEnergy(const double& aValue)
{
    fMinEnergy = aValue;
}

}  // namespace Kassiopeia

#endif
