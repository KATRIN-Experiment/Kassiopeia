#ifndef Kassiopeia_KSGenValueRadiusFraction_h_
#define Kassiopeia_KSGenValueRadiusFraction_h_

#include "KField.h"
#include "KSGenValue.h"

namespace Kassiopeia
{
class KSGenValueRadiusFraction : public KSComponentTemplate<KSGenValueRadiusFraction, KSGenValue>
{
  public:
    KSGenValueRadiusFraction();
    KSGenValueRadiusFraction(const KSGenValueRadiusFraction& aCopy);
    KSGenValueRadiusFraction* Clone() const override;
    ~KSGenValueRadiusFraction() override;

  public:
    void DiceValue(std::vector<double>& aDicedValues) override;
};

}  // namespace Kassiopeia

#endif
