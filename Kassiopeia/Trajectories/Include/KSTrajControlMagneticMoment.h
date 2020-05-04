#ifndef Kassiopeia_KSTrajControlMagneticMoment_h_
#define Kassiopeia_KSTrajControlMagneticMoment_h_

#include "KSComponentTemplate.h"
#include "KSTrajAdiabaticSpinTypes.h"
#include "KSTrajAdiabaticTypes.h"
#include "KSTrajExactSpinTypes.h"
#include "KSTrajExactTypes.h"

namespace Kassiopeia
{

class KSTrajControlMagneticMoment :
    public KSComponentTemplate<KSTrajControlMagneticMoment>,
    public KSTrajExactControl,
    public KSTrajExactSpinControl,
    public KSTrajAdiabaticSpinControl,
    public KSTrajAdiabaticControl
{
  public:
    KSTrajControlMagneticMoment();
    KSTrajControlMagneticMoment(const KSTrajControlMagneticMoment& aCopy);
    KSTrajControlMagneticMoment* Clone() const override;
    ~KSTrajControlMagneticMoment() override;

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
    void SetLowerLimit(const double& aLowerLimit);
    void SetUpperLimit(const double& aUpperLimit);

  protected:
    virtual void ActivateObject();

  private:
    double fLowerLimit;
    double fUpperLimit;
    double fTimeStep;
    bool fFirstStep;
};

inline void KSTrajControlMagneticMoment::SetLowerLimit(const double& aLowerLimit)
{
    fLowerLimit = aLowerLimit;
    return;
}
inline void KSTrajControlMagneticMoment::SetUpperLimit(const double& aUpperLimit)
{
    fUpperLimit = aUpperLimit;
    return;
}

}  // namespace Kassiopeia

#endif
