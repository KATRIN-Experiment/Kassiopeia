#ifndef Kassiopeia_KSIntDecayCalculator_h_
#define Kassiopeia_KSIntDecayCalculator_h_

#include "KField.h"
#include "KSComponentTemplate.h"
#include "KSGenerator.h"
#include "KSParticle.h"

namespace Kassiopeia
{

class KSIntDecayCalculator : public KSComponentTemplate<KSIntDecayCalculator>
{
  public:
    KSIntDecayCalculator();
    ~KSIntDecayCalculator() override;
    KSIntDecayCalculator* Clone() const override = 0;

  public:
    virtual void CalculateLifeTime(const KSParticle& aParticle, double& aCrossSection) = 0;
    virtual void ExecuteInteraction(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                                    KSParticleQueue& aSecondaries) = 0;

  protected:
    void PullDeupdateComponent() override;
    void PushDeupdateComponent() override;


    //variables for output
    K_REFS(int, StepNDecays)
    K_REFS(double, StepEnergyLoss)
};

}  // namespace Kassiopeia

#endif
