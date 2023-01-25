#ifndef Kassiopeia_KSGenValueFermi_h_
#define Kassiopeia_KSGenValueFermi_h_

#include "KField.h"
#include "KMathBracketingSolver.h"
#include "KSGenValue.h"

namespace Kassiopeia
{
class KSGenValueFermi : public KSComponentTemplate<KSGenValueFermi, KSGenValue>
{
  public:
    KSGenValueFermi();
    KSGenValueFermi(const KSGenValueFermi& aCopy);
    KSGenValueFermi* Clone() const override;
    ~KSGenValueFermi() override;

  public:
    void DiceValue(std::vector<double>& aDicedValues) override;

  public:
    K_SET_GET(double, ValueMin)
    K_SET_GET(double, ValueMax)
    K_SET_GET(double, ValueMean)
    K_SET_GET(double, ValueTau)
    K_SET_GET(double, ValueTemp)

  protected:
    double ValueFunction(const double& aValue) const;
    katrin::KMathBracketingSolver fSolver;
};

}  // namespace Kassiopeia

#endif
