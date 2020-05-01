#ifndef Kassiopeia_KSTermStepsize_h_
#define Kassiopeia_KSTermStepsize_h_

#include "KField.h"
#include "KSTerminator.h"

namespace Kassiopeia
{

class KSParticle;

class KSTermStepsize : public KSComponentTemplate<KSTermStepsize, KSTerminator>
{
  public:
    KSTermStepsize();
    KSTermStepsize(const KSTermStepsize& aCopy);
    KSTermStepsize* Clone() const override;
    ~KSTermStepsize() override;

  public:
    void CalculateTermination(const KSParticle& anInitialParticle, bool& aFlag) override;
    void ExecuteTermination(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                            KSParticleQueue& aParticleQueue) const override;

  public:
    ;
    K_SET_GET(double, LowerLimit);
    K_SET_GET(double, UpperLimit)

  private:
    double fCurrentPathLength;
};


}  // namespace Kassiopeia

#endif
