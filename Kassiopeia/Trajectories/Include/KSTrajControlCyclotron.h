#ifndef Kassiopeia_KSTrajControlCyclotron_h_
#define Kassiopeia_KSTrajControlCyclotron_h_

#include "KSComponentTemplate.h"
#include "KSTrajAdiabaticSpinTypes.h"
#include "KSTrajAdiabaticTypes.h"
#include "KSTrajExactSpinTypes.h"
#include "KSTrajExactTypes.h"

namespace Kassiopeia
{

class KSTrajControlCyclotron :
    public KSComponentTemplate<KSTrajControlCyclotron>,
    public KSTrajExactControl,
    public KSTrajExactSpinControl,
    public KSTrajAdiabaticSpinControl,
    public KSTrajAdiabaticControl
{
  public:
    KSTrajControlCyclotron();
    KSTrajControlCyclotron(const KSTrajControlCyclotron& aCopy);
    KSTrajControlCyclotron* Clone() const override;
    ~KSTrajControlCyclotron() override;

  public:
    void Calculate(const KSTrajExactParticle& aParticle, double& aValue) override;
    void Check(const KSTrajExactParticle& anInitialParticle, const KSTrajExactParticle& aFinalParticle,
               const KSTrajExactError& anError, bool& aFlag) override;

    void Calculate(const KSTrajExactSpinParticle& aParticle, double& aValue) override;
    void Check(const KSTrajExactSpinParticle& anInitialParticle, const KSTrajExactSpinParticle& aFinalParticle,
               const KSTrajExactSpinError& anError, bool& aFlag) override;

    void Calculate(const KSTrajAdiabaticSpinParticle& aParticle, double& aValue) override;
    void Check(const KSTrajAdiabaticSpinParticle& anInitialParticle, const KSTrajAdiabaticSpinParticle& aFinalParticle,
               const KSTrajAdiabaticSpinError& anError, bool& aFlag) override;

    void Calculate(const KSTrajAdiabaticParticle& aParticle, double& aValue) override;
    void Check(const KSTrajAdiabaticParticle& anInitialParticle, const KSTrajAdiabaticParticle& aFinalParticle,
               const KSTrajAdiabaticError& anError, bool& aFlag) override;

  public:
    void SetFraction(const double& aFraction);

  private:
    double fFraction;
};

inline void KSTrajControlCyclotron::SetFraction(const double& aFraction)
{
    fFraction = aFraction;
    return;
}

}  // namespace Kassiopeia

#endif
