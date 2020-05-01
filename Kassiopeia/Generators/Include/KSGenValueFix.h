#ifndef Kassiopeia_KSGenValueFix_h_
#define Kassiopeia_KSGenValueFix_h_

#include "KField.h"
#include "KSGenValue.h"

namespace Kassiopeia
{

class KSGenValueFix : public KSComponentTemplate<KSGenValueFix, KSGenValue>
{
  public:
    KSGenValueFix();
    KSGenValueFix(const KSGenValueFix& aCopy);
    KSGenValueFix* Clone() const override;
    ~KSGenValueFix() override;

  public:
    void DiceValue(std::vector<double>& aDicedValues) override;

  public:
    K_SET_GET(double, Value)
};

}  // namespace Kassiopeia

#endif
