//
// Created by trost on 03.06.15.
//

#ifndef KASPER_KSINTDECAYCALCULATORFERENCIONISATION_H
#define KASPER_KSINTDECAYCALCULATORFERENCIONISATION_H


#include "KField.h"
#include "KSIntDecayCalculator.h"
#include "RydbergFerenc.h"


namespace Kassiopeia
{

class KSIntDecayCalculatorFerencIonisation :
    public KSComponentTemplate<KSIntDecayCalculatorFerencIonisation, KSIntDecayCalculator>
{

  public:
    KSIntDecayCalculatorFerencIonisation();
    ~KSIntDecayCalculatorFerencIonisation() override;

    KSIntDecayCalculatorFerencIonisation(const KSIntDecayCalculatorFerencIonisation& aCopy);

    KSIntDecayCalculatorFerencIonisation* Clone() const override;

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
    K_SET_GET(double, Temperature)

  public:
    void SetDecayProductGenerator(KSGenerator* const aGenerator);

    KSGenerator* GetDecayProductGenerator() const;

  protected:
    KSGenerator* fDecayProductGenerator;

  private:
    double low_n_lifetimes[150][150];

  private:
    RydbergCalculator* fCalculator;
};
}  // namespace Kassiopeia


#endif  //KASPER_KSINTDECAYCALCULATORFERENCIONISATION_H