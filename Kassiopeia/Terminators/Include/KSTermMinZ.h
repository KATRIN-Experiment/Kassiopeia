#ifndef Kassiopeia_KSTermMinZ_h_
#define Kassiopeia_KSTermMinZ_h_

#include "KSTerminator.h"

namespace Kassiopeia
{

class KSParticle;

class KSTermMinZ : public KSComponentTemplate<KSTermMinZ, KSTerminator>
{
  public:
    KSTermMinZ();
    KSTermMinZ(const KSTermMinZ& aCopy);
    KSTermMinZ* Clone() const override;
    ~KSTermMinZ() override;

  public:
    void CalculateTermination(const KSParticle& anInitialParticle, bool& aFlag) override;
    void ExecuteTermination(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                            KSParticleQueue& aParticleQueue) const override;

  public:
    void SetMinZ(const double& aValue);

  private:
    double fMinZ;
};

inline void KSTermMinZ::SetMinZ(const double& aValue)
{
    fMinZ = aValue;
}

}  // namespace Kassiopeia

#endif
