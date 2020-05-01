#ifndef Kassiopeia_KSIntDensityConstant_h_
#define Kassiopeia_KSIntDensityConstant_h_

#include "KField.h"
#include "KSIntDensity.h"

namespace Kassiopeia
{
class KSIntDensityConstant : public KSComponentTemplate<KSIntDensityConstant, KSIntDensity>
{
  public:
    KSIntDensityConstant();
    KSIntDensityConstant(const KSIntDensityConstant& aCopy);
    KSIntDensityConstant* Clone() const override;
    ~KSIntDensityConstant() override;

  public:
    void CalculateDensity(const KSParticle& aParticle, double& aDensity) override;

  public:
    K_SET_GET(double, Temperature)  // kelvin
    K_SET_GET(double, Pressure)     // pascal (SI UNITS!)
    K_SET_GET(double, Density)      // m^-3
};

}  // namespace Kassiopeia

#endif
