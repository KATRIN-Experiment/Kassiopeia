#ifndef Kassiopeia_KSGenValueAngleSpherical_h_
#define Kassiopeia_KSGenValueAngleSpherical_h_

#include "KField.h"
#include "KSGenValue.h"

namespace Kassiopeia
{
class KSGenValueAngleSpherical : public KSComponentTemplate<KSGenValueAngleSpherical, KSGenValue>
{
  public:
    KSGenValueAngleSpherical();
    KSGenValueAngleSpherical(const KSGenValueAngleSpherical& aCopy);
    KSGenValueAngleSpherical* Clone() const override;
    ~KSGenValueAngleSpherical() override;

  public:
    void DiceValue(std::vector<double>& aDicedValues) override;

  public:
    K_SET_GET(double, AngleMin)
    K_SET_GET(double, AngleMax)
};

}  // namespace Kassiopeia

#endif
