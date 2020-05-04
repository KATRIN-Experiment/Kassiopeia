#ifndef Kassiopeia_KSIntDecayCalculatorGlukhovSpontaneous_h_
#define Kassiopeia_KSIntDecayCalculatorGlukhovSpontaneous_h_

#include "KField.h"
#include "KSIntDecayCalculator.h"

namespace Kassiopeia
{
class KSIntDecayCalculatorGlukhovSpontaneous :
    public KSComponentTemplate<KSIntDecayCalculatorGlukhovSpontaneous, KSIntDecayCalculator>
{
  public:
    KSIntDecayCalculatorGlukhovSpontaneous();
    KSIntDecayCalculatorGlukhovSpontaneous(const KSIntDecayCalculatorGlukhovSpontaneous& aCopy);
    KSIntDecayCalculatorGlukhovSpontaneous* Clone() const override;
    ~KSIntDecayCalculatorGlukhovSpontaneous() override;

  public:
    void CalculateLifeTime(const KSParticle& aParticle, double& aLifeTime) override;
    void ExecuteInteraction(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                            KSParticleQueue& aSecondaries) override;


  public:
    K_SET_GET(long long, TargetPID)
    K_SET_GET(long long, minPID)
    K_SET_GET(long long, maxPID)

  private:
    static const double p_coefficients[3][4];

  public:
    static double CalculateSpontaneousDecayRate(int n, int l);
};


}  // namespace Kassiopeia

#endif
