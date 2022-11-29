#ifndef Kassiopeia_KSTermTrapped_h_
#define Kassiopeia_KSTermTrapped_h_

#include "KField.h"
#include "KSTerminator.h"

namespace Kassiopeia
{

class KSParticle;

class KSTermTrapped : public KSComponentTemplate<KSTermTrapped, KSTerminator>
{
  public:
    KSTermTrapped();
    KSTermTrapped(const KSTermTrapped& aCopy);
    KSTermTrapped* Clone() const override;
    ~KSTermTrapped() override;

    void SetUseMagneticField(bool aFlag);
    void SetUseElectricField(bool aFlag);

    K_GET(int, UseMagneticField)
    K_GET(int, UseElectricField)
    K_SET_GET(int, MaxTurns)

  public:
    void CalculateTermination(const KSParticle& anInitialParticle, bool& aFlag) override;
    void ExecuteTermination(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                            KSParticleQueue& aParticleQueue) const override;

  protected:
    void ActivateComponent() override;
    void DeactivateComponent() override;

  private:
    int fCurrentTurns;
    double fCurrentDotProduct;
};

}  // namespace Kassiopeia

#endif
