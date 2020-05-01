#ifndef Kassiopeia_KSGenValueUniform_h_
#define Kassiopeia_KSGenValueUniform_h_

#include "KField.h"
#include "KSGenValue.h"

namespace Kassiopeia
{

class KSGenValueUniform : public KSComponentTemplate<KSGenValueUniform, KSGenValue>
{
  public:
    KSGenValueUniform();
    KSGenValueUniform(const KSGenValueUniform& aCopy);
    KSGenValueUniform* Clone() const override;
    ~KSGenValueUniform() override;

  public:
    void DiceValue(std::vector<double>& aDicedValues) override;

  public:
    K_SET_GET(double, ValueMin)
    K_SET_GET(double, ValueMax)
};

}  // namespace Kassiopeia

#endif
