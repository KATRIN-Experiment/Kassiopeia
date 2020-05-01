#ifndef Kassiopeia_KSTrajTermDrift_h_
#define Kassiopeia_KSTrajTermDrift_h_

#include "KSComponentTemplate.h"
#include "KSTrajAdiabaticTypes.h"

namespace Kassiopeia
{

class KSTrajTermDrift : public KSComponentTemplate<KSTrajTermDrift>, public KSTrajAdiabaticDifferentiator
{
  public:
    KSTrajTermDrift();
    KSTrajTermDrift(const KSTrajTermDrift& aCopy);
    KSTrajTermDrift* Clone() const override;
    ~KSTrajTermDrift() override;

  public:
    void Differentiate(double /*aTime*/, const KSTrajAdiabaticParticle& aParticle,
                       KSTrajAdiabaticDerivative& aDerivative) const override;

    const KThreeVector& GetDriftVelocity() const
    {
        return fDriftVelocity;
    }
    const double& GetLongitudinalForce() const
    {
        return fLongitudinalForce;
    }
    const double& GetTransverseForce() const
    {
        return fTransverseForce;
    }

  private:
    mutable KThreeVector fDriftVelocity;
    mutable double fLongitudinalForce;
    mutable double fTransverseForce;
};

}  // namespace Kassiopeia

#endif
