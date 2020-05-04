//
// Created by trost on 27.05.15.
//

#ifndef KASPER_KSINTDECAYCALCULATORFERENCSPONTANEOUS_H
#define KASPER_KSINTDECAYCALCULATORFERENCSPONTANEOUS_H

#include "KField.h"
#include "KSIntDecayCalculator.h"
#include "RydbergFerenc.h"

namespace Kassiopeia
{
class KSIntDecayCalculatorFerencSpontaneous :
    public KSComponentTemplate<KSIntDecayCalculatorFerencSpontaneous, KSIntDecayCalculator>
{
  public:
    KSIntDecayCalculatorFerencSpontaneous();
    KSIntDecayCalculatorFerencSpontaneous(const KSIntDecayCalculatorFerencSpontaneous& aCopy);
    KSIntDecayCalculatorFerencSpontaneous* Clone() const override;
    ~KSIntDecayCalculatorFerencSpontaneous() override;

  public:
    void CalculateLifeTime(const KSParticle& aParticle, double& aLifeTime) override;
    void ExecuteInteraction(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                            KSParticleQueue& aSecondaries) override;

  protected:
    void InitializeComponent() override;

  public:
    K_SET_GET(long long, TargetPID)
    K_SET_GET(long long, minPID)
    K_SET_GET(long long, maxPID)

  private:
    int fLastn;
    int fLastl;
    double fLastLifetime;

  private:
    RydbergCalculator* fCalculator;

  private:
    double low_n_lifetimes[150][150];
};


}  // namespace Kassiopeia

#endif  //KASPER_KSINTDECAYCALCULATORFERENCSPONTANEOUS_H
