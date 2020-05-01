#ifndef Kassiopeia_KSTrajTermGyration_h_
#define Kassiopeia_KSTrajTermGyration_h_

#include "KSComponentTemplate.h"
#include "KSTrajAdiabaticTypes.h"

namespace Kassiopeia
{

class KSTrajTermGyration : public KSComponentTemplate<KSTrajTermGyration>, public KSTrajAdiabaticDifferentiator
{
  public:
    KSTrajTermGyration();
    KSTrajTermGyration(const KSTrajTermGyration& aCopy);
    KSTrajTermGyration* Clone() const override;
    ~KSTrajTermGyration() override;

  public:
    void Differentiate(double /*aTime*/, const KSTrajAdiabaticParticle& aParticle,
                       KSTrajAdiabaticDerivative& aDerivative) const override;

    const double& GetPhaseVelocity() const
    {
        return fPhaseVelocity;
    }

  private:
    mutable double fPhaseVelocity;
};

}  // namespace Kassiopeia

#endif
