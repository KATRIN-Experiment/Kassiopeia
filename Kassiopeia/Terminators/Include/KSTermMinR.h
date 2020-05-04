#ifndef Kassiopeia_KSTermMinR_h_
#define Kassiopeia_KSTermMinR_h_

#include "KSTerminator.h"

namespace Kassiopeia
{

class KSParticle;

class KSTermMinR : public KSComponentTemplate<KSTermMinR, KSTerminator>
{
  public:
    KSTermMinR();
    KSTermMinR(const KSTermMinR& aCopy);
    KSTermMinR* Clone() const override;
    ~KSTermMinR() override;

  public:
    void CalculateTermination(const KSParticle& anInitialParticle, bool& aFlag) override;
    void ExecuteTermination(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                            KSParticleQueue& aParticleQueue) const override;

  public:
    void SetMinR(const double& aValue);

  private:
    double fMinR;
};

inline void KSTermMinR::SetMinR(const double& aValue)
{
    fMinR = aValue;
}

}  // namespace Kassiopeia

#endif
