#ifndef Kassiopeia_KSTrajControlSpinPrecession_h_
#define Kassiopeia_KSTrajControlSpinPrecession_h_

#include "KSComponentTemplate.h"
#include "KSTrajAdiabaticSpinTypes.h"
#include "KSTrajExactSpinTypes.h"

namespace Kassiopeia
{

class KSTrajControlSpinPrecession :
    public KSComponentTemplate<KSTrajControlSpinPrecession>,
    public KSTrajExactSpinControl,
    public KSTrajAdiabaticSpinControl
{
  public:
    KSTrajControlSpinPrecession();
    KSTrajControlSpinPrecession(const KSTrajControlSpinPrecession& aCopy);
    KSTrajControlSpinPrecession* Clone() const override;
    ~KSTrajControlSpinPrecession() override;

  public:
    void Calculate(const KSTrajExactSpinParticle& aParticle, double& aValue) override;
    void Check(const KSTrajExactSpinParticle& anInitialParticle, const KSTrajExactSpinParticle& aFinalParticle,
               const KSTrajExactSpinError& anError, bool& aFlag) override;

    void Calculate(const KSTrajAdiabaticSpinParticle& aParticle, double& aValue) override;
    void Check(const KSTrajAdiabaticSpinParticle& anInitialParticle, const KSTrajAdiabaticSpinParticle& aFinalParticle,
               const KSTrajAdiabaticSpinError& anError, bool& aFlag) override;

  public:
    void SetFraction(const double& aFraction);

  private:
    double fFraction;
};

inline void KSTrajControlSpinPrecession::SetFraction(const double& aFraction)
{
    fFraction = aFraction;
    return;
}

}  // namespace Kassiopeia

#endif
