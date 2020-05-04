#ifndef Kassiopeia_KSTrajControlTime_h_
#define Kassiopeia_KSTrajControlTime_h_

#include "KSComponentTemplate.h"
#include "KSTrajAdiabaticSpinTypes.h"
#include "KSTrajAdiabaticTypes.h"
#include "KSTrajElectricTypes.h"
#include "KSTrajExactSpinTypes.h"
#include "KSTrajExactTrappedTypes.h"
#include "KSTrajExactTypes.h"
#include "KSTrajMagneticTypes.h"

namespace Kassiopeia
{

class KSTrajControlTime :
    public KSComponentTemplate<KSTrajControlTime>,
    public KSTrajExactControl,
    public KSTrajExactSpinControl,
    public KSTrajExactTrappedControl,
    public KSTrajAdiabaticControl,
    public KSTrajAdiabaticSpinControl,
    public KSTrajElectricControl,
    public KSTrajMagneticControl
{
  public:
    KSTrajControlTime();
    KSTrajControlTime(const KSTrajControlTime& aCopy);
    KSTrajControlTime* Clone() const override;
    ~KSTrajControlTime() override;

  public:
    void Calculate(const KSTrajExactParticle& aParticle, double& aValue) override;
    void Check(const KSTrajExactParticle& anInitialParticle, const KSTrajExactParticle& aFinalParticle,
               const KSTrajExactError& anError, bool& aFlag) override;

    void Calculate(const KSTrajExactSpinParticle& aParticle, double& aValue) override;
    void Check(const KSTrajExactSpinParticle& anInitialParticle, const KSTrajExactSpinParticle& aFinalParticle,
               const KSTrajExactSpinError& anError, bool& aFlag) override;

    void Calculate(const KSTrajExactTrappedParticle& aParticle, double& aValue) override;
    void Check(const KSTrajExactTrappedParticle& anInitialParticle, const KSTrajExactTrappedParticle& aFinalParticle,
               const KSTrajExactTrappedError& anError, bool& aFlag) override;

    void Calculate(const KSTrajAdiabaticParticle& aParticle, double& aValue) override;
    void Check(const KSTrajAdiabaticParticle& anInitialParticle, const KSTrajAdiabaticParticle& aFinalParticle,
               const KSTrajAdiabaticError& anError, bool& aFlag) override;

    void Calculate(const KSTrajAdiabaticSpinParticle& aParticle, double& aValue) override;
    void Check(const KSTrajAdiabaticSpinParticle& anInitialParticle, const KSTrajAdiabaticSpinParticle& aFinalParticle,
               const KSTrajAdiabaticSpinError& anError, bool& aFlag) override;

    void Calculate(const KSTrajElectricParticle& aParticle, double& aValue) override;
    void Check(const KSTrajElectricParticle& anInitialParticle, const KSTrajElectricParticle& aFinalParticle,
               const KSTrajElectricError& anError, bool& aFlag) override;

    void Calculate(const KSTrajMagneticParticle& aParticle, double& aValue) override;
    void Check(const KSTrajMagneticParticle& anInitialParticle, const KSTrajMagneticParticle& aFinalParticle,
               const KSTrajMagneticError& anError, bool& aFlag) override;

  public:
    void SetTime(const double& aTime);

  private:
    double fTime;
};

inline void KSTrajControlTime::SetTime(const double& aTime)
{
    fTime = aTime;
    return;
}

}  // namespace Kassiopeia

#endif
