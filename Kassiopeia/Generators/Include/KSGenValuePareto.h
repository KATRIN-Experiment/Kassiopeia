#ifndef Kassiopeia_KSGenValuePareto_h_
#define Kassiopeia_KSGenValuePareto_h_

#include "KField.h"
#include "KMathBracketingSolver.h"
#include "KSGenValue.h"

namespace Kassiopeia
{
class KSGenValuePareto : public KSComponentTemplate<KSGenValuePareto, KSGenValue>
{
  public:
    KSGenValuePareto();
    KSGenValuePareto(const KSGenValuePareto& aCopy);
    KSGenValuePareto* Clone() const override;
    ~KSGenValuePareto() override;

  public:
    void DiceValue(std::vector<double>& aDicedValues) override;

  public:
    K_SET_GET(double, Slope)
    K_SET_GET(double, Cutoff)
    K_SET_GET(double, Offset)
    K_SET_GET(double, ValueMin)
    K_SET_GET(double, ValueMax)

  public:
    void InitializeComponent() override;
    void DeinitializeComponent() override;

  protected:
    double fValueParetoMin;
    double fValueParetoMax;
};

}  // namespace Kassiopeia

#endif
