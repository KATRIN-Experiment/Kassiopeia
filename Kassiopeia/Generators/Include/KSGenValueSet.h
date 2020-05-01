#ifndef Kassiopeia_KSGenValueSet_h_
#define Kassiopeia_KSGenValueSet_h_

#include "KField.h"
#include "KSGenValue.h"

namespace Kassiopeia
{

class KSGenValueSet : public KSComponentTemplate<KSGenValueSet, KSGenValue>
{
  public:
    KSGenValueSet();
    KSGenValueSet(const KSGenValueSet& aCopy);
    KSGenValueSet* Clone() const override;
    ~KSGenValueSet() override;

  public:
    void DiceValue(std::vector<double>& aDicedValues) override;

  public:
    K_SET_GET(double, ValueStart)
    K_SET_GET(double, ValueStop)
    K_SET_GET(double, ValueIncrement)
    K_SET_GET(unsigned int, ValueCount)
};

}  // namespace Kassiopeia

#endif
