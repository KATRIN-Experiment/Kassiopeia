#ifndef Kassiopeia_KSGenValueGeneralizedGauss_h_
#define Kassiopeia_KSGenValueGeneralizedGauss_h_

#include "KField.h"
#include "KMathBracketingSolver.h"
#include "KSGenValue.h"

namespace Kassiopeia
{
class KSGenValueGeneralizedGauss : public KSComponentTemplate<KSGenValueGeneralizedGauss, KSGenValue>
{
  public:
    KSGenValueGeneralizedGauss();
    KSGenValueGeneralizedGauss(const KSGenValueGeneralizedGauss& aCopy);
    KSGenValueGeneralizedGauss* Clone() const override;
    ~KSGenValueGeneralizedGauss() override;

  public:
    void DiceValue(std::vector<double>& aDicedValues) override;

  public:
    K_SET_GET(double, ValueMin)
    K_SET_GET(double, ValueMax)
    K_SET_GET(double, ValueMean)
    K_SET_GET(double, ValueSigma)
    K_SET_GET(double, ValueSkew)

  protected:
    double ValueFunction(const double& aValue) const;
    katrin::KMathBracketingSolver fSolver;
};

}  // namespace Kassiopeia

#endif
