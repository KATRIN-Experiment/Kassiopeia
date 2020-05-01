#ifndef Kassiopeia_KSGenValueBoltzmann_h_
#define Kassiopeia_KSGenValueBoltzmann_h_

#include "KField.h"
#include "KSGenValue.h"
// #include "KMathBracketingSolver.h"
// using katrin::KMathBracketingSolver;

namespace Kassiopeia
{
class KSGenValueBoltzmann : public KSComponentTemplate<KSGenValueBoltzmann, KSGenValue>
{
  public:
    KSGenValueBoltzmann();
    KSGenValueBoltzmann(const KSGenValueBoltzmann& aCopy);
    KSGenValueBoltzmann* Clone() const override;
    ~KSGenValueBoltzmann() override;

  public:
    void DiceValue(std::vector<double>& aDicedValues) override;

  public:
    K_SET_GET(double, ValueMass)
    K_SET_GET(double, ValuekT)
    K_SET_GET(bool, UseElectronVolts)
};

}  // namespace Kassiopeia

#endif
