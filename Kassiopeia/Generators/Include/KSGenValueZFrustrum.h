#ifndef Kassiopeia_KSGenValueZFrustrum_h_
#define Kassiopeia_KSGenValueZFrustrum_h_

#include "KField.h"
#include "KSGenValue.h"

namespace Kassiopeia
{
class KSGenValueZFrustrum : public KSComponentTemplate<KSGenValueZFrustrum, KSGenValue>
{
  public:
    KSGenValueZFrustrum();
    KSGenValueZFrustrum(const KSGenValueZFrustrum& aCopy);
    KSGenValueZFrustrum* Clone() const override;
    ~KSGenValueZFrustrum() override;

  public:
    void DiceValue(std::vector<double>& aDicedValues) override;

  public:
    K_SET_GET(double, r1)
    K_SET_GET(double, r2)
    K_SET_GET(double, z1)
    K_SET_GET(double, z2)
};

}  // namespace Kassiopeia

#endif
