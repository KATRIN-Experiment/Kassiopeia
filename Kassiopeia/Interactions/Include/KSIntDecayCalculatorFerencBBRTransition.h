//
// Created by trost on 27.05.15.
//

#ifndef KASPER_KSINTDECAYCALCULATORFERENCBBRTRANSITION_H
#define KASPER_KSINTDECAYCALCULATORFERENCBBRTRANSITION_H

#include "KField.h"
#include "KSIntDecayCalculator.h"
#include "RydbergFerenc.h"

namespace Kassiopeia
{
class KSIntDecayCalculatorFerencBBRTransition :
    public KSComponentTemplate<KSIntDecayCalculatorFerencBBRTransition, KSIntDecayCalculator>
{
  public:
    KSIntDecayCalculatorFerencBBRTransition();
    KSIntDecayCalculatorFerencBBRTransition(const KSIntDecayCalculatorFerencBBRTransition& aCopy);
    KSIntDecayCalculatorFerencBBRTransition* Clone() const override;
    ~KSIntDecayCalculatorFerencBBRTransition() override;

  public:
    void CalculateLifeTime(const KSParticle& aParticle, double& aLifeTime) override;
    void ExecuteInteraction(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                            KSParticleQueue& aSecondaries) override;


  public:
    K_SET_GET(long long, TargetPID)
    K_SET_GET(long long, minPID)
    K_SET_GET(long long, maxPID)
    K_SET_GET(double, Temperature)

  protected:
    void InitializeComponent() override;

  private:
    int fLastn;
    int fLastl;
    double fLastLifetime;
    double low_n_lifetimes[150][150];

  private:
    RydbergCalculator* fCalculator;
};


}  // namespace Kassiopeia


#endif  //KASPER_KSINTDECAYCALCULATORFERENCBBRTRANSITION_H
