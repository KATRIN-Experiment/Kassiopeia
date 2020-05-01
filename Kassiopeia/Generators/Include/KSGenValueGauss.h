#ifndef Kassiopeia_KSGenValueGauss_h_
#define Kassiopeia_KSGenValueGauss_h_

#include "KField.h"
#include "KMathBracketingSolver.h"
#include "KSGenValue.h"
using katrin::KMathBracketingSolver;

namespace Kassiopeia
{
class KSGenValueGauss : public KSComponentTemplate<KSGenValueGauss, KSGenValue>
{
  public:
    KSGenValueGauss();
    KSGenValueGauss(const KSGenValueGauss& aCopy);
    KSGenValueGauss* Clone() const override;
    ~KSGenValueGauss() override;

  public:
    void DiceValue(std::vector<double>& aDicedValues) override;

  public:
    K_SET_GET(double, ValueMin)
    K_SET_GET(double, ValueMax)
    K_SET_GET(double, ValueMean)
    K_SET_GET(double, ValueSigma)

  protected:
    double ValueFunction(const double& aValue) const;
    KMathBracketingSolver fSolver;
};

}  // namespace Kassiopeia

#endif
