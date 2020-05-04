#ifndef Kassiopeia_KSTrajTermSynchrotron_h_
#define Kassiopeia_KSTrajTermSynchrotron_h_

#include "KSComponentTemplate.h"
#include "KSTrajAdiabaticTypes.h"
#include "KSTrajExactTrappedTypes.h"
#include "KSTrajExactTypes.h"

namespace Kassiopeia
{

class KSTrajTermSynchrotron :
    public KSComponentTemplate<KSTrajTermSynchrotron>,
    public KSTrajExactDifferentiator,
    public KSTrajAdiabaticDifferentiator,
    public KSTrajExactTrappedDifferentiator
{
  public:
    KSTrajTermSynchrotron();
    KSTrajTermSynchrotron(const KSTrajTermSynchrotron& aCopy);
    KSTrajTermSynchrotron* Clone() const override;
    ~KSTrajTermSynchrotron() override;

  public:
    void Differentiate(double /*aTime*/, const KSTrajExactParticle& aParticle,
                       KSTrajExactDerivative& aDerivative) const override;
    void Differentiate(double /*aTime*/, const KSTrajAdiabaticParticle& aParticle,
                       KSTrajAdiabaticDerivative& aDerivative) const override;
    void Differentiate(double aTime, const KSTrajExactTrappedParticle& aParticle,
                       KSTrajExactTrappedDerivative& aDerivative) const override;

  public:
    void SetEnhancement(const double& anEnhancement);
    void SetOldMethode(const bool& aBool);

    const double& GetTotalForce() const
    {
        return fTotalForce;
    }

  private:
    double fEnhancement;
    bool fOldMethode;

    mutable double fTotalForce;
};

}  // namespace Kassiopeia

#endif
