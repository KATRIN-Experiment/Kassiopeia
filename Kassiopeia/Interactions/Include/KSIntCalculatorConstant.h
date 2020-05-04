#ifndef Kassiopeia_KSIntCalculatorConstant_h_
#define Kassiopeia_KSIntCalculatorConstant_h_

#include "KField.h"
#include "KSIntCalculator.h"

namespace Kassiopeia
{
class KSIntCalculatorConstant : public KSComponentTemplate<KSIntCalculatorConstant, KSIntCalculator>
{
  public:
    KSIntCalculatorConstant();
    KSIntCalculatorConstant(const KSIntCalculatorConstant& aCopy);
    KSIntCalculatorConstant* Clone() const override;
    ~KSIntCalculatorConstant() override;

  public:
    void CalculateCrossSection(const KSParticle& aParticle, double& aCrossSection) override;
    void ExecuteInteraction(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                            KSParticleQueue& aSecondaries) override;

  public:
    K_SET_GET(double, CrossSection)  // m^2
};

}  // namespace Kassiopeia

#endif
