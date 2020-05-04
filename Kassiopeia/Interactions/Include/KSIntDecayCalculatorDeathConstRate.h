#ifndef Kassiopeia_KSIntDecayCalculatorDeathConstRate_h_
#define Kassiopeia_KSIntDecayCalculatorDeathConstRate_h_

#include "KField.h"
#include "KSIntDecayCalculator.h"

namespace Kassiopeia
{
class KSIntDecayCalculatorDeathConstRate :
    public KSComponentTemplate<KSIntDecayCalculatorDeathConstRate, KSIntDecayCalculator>
{
  public:
    KSIntDecayCalculatorDeathConstRate();
    KSIntDecayCalculatorDeathConstRate(const KSIntDecayCalculatorDeathConstRate& aCopy);
    KSIntDecayCalculatorDeathConstRate* Clone() const override;
    ~KSIntDecayCalculatorDeathConstRate() override;

  public:
    void CalculateLifeTime(const KSParticle& aParticle, double& aLifeTime) override;
    void ExecuteInteraction(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                            KSParticleQueue& aSecondaries) override;


  public:
    K_SET_GET(double, LifeTime)  // s
    K_SET_GET(long long, TargetPID)
    K_SET_GET(long long, minPID)
    K_SET_GET(long long, maxPID)

  public:
    void SetDecayProductGenerator(KSGenerator* const aGenerator);
    KSGenerator* GetDecayProductGenerator() const;

  protected:
    KSGenerator* fDecayProductGenerator;
};


}  // namespace Kassiopeia

#endif
