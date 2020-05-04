#ifndef Kassiopeia_KSGenValueRadiusSpherical_h_
#define Kassiopeia_KSGenValueRadiusSpherical_h_

#include "KField.h"
#include "KSGenValue.h"

namespace Kassiopeia
{
class KSGenValueRadiusSpherical : public KSComponentTemplate<KSGenValueRadiusSpherical, KSGenValue>
{
  public:
    KSGenValueRadiusSpherical();
    KSGenValueRadiusSpherical(const KSGenValueRadiusSpherical& aCopy);
    KSGenValueRadiusSpherical* Clone() const override;
    ~KSGenValueRadiusSpherical() override;

  public:
    void DiceValue(std::vector<double>& aDicedValues) override;

  public:
    K_SET_GET(double, RadiusMin)
    K_SET_GET(double, RadiusMax)
};

}  // namespace Kassiopeia

#endif
