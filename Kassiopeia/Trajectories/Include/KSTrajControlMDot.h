#ifndef Kassiopeia_KSTrajControlMDot_h_
#define Kassiopeia_KSTrajControlMDot_h_

#include "KSComponentTemplate.h"
#include "KSTrajAdiabaticSpinTypes.h"

namespace Kassiopeia
{

class KSTrajControlMDot : public KSComponentTemplate<KSTrajControlMDot>, public KSTrajAdiabaticSpinControl
{
  public:
    KSTrajControlMDot();
    KSTrajControlMDot(const KSTrajControlMDot& aCopy);
    KSTrajControlMDot* Clone() const override;
    ~KSTrajControlMDot() override;

  public:
    void Calculate(const KSTrajAdiabaticSpinParticle& aParticle, double& aValue) override;
    void Check(const KSTrajAdiabaticSpinParticle& anInitialParticle, const KSTrajAdiabaticSpinParticle& aFinalParticle,
               const KSTrajAdiabaticSpinError& anError, bool& aFlag) override;

  public:
    void SetFraction(const double& aFraction);

  private:
    double fFraction;
};

inline void KSTrajControlMDot::SetFraction(const double& aFraction)
{
    fFraction = aFraction;
    return;
}

}  // namespace Kassiopeia

#endif
