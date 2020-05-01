#ifndef Kassiopeia_KSIntDensity_h_
#define Kassiopeia_KSIntDensity_h_

#include "KSComponentTemplate.h"
#include "KSParticle.h"

namespace Kassiopeia
{
class KSStep;

class KSIntDensity : public KSComponentTemplate<KSIntDensity>
{
  public:
    KSIntDensity();
    ~KSIntDensity() override;
    KSIntDensity* Clone() const override = 0;

  public:
    virtual void CalculateDensity(const KSParticle& aParticle, double& aDensity) = 0;
};

}  // namespace Kassiopeia

#endif
