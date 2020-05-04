#ifndef Kassiopeia_KSTrajControlLength_h_
#define Kassiopeia_KSTrajControlLength_h_

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

class KSTrajControlLength :
    public KSComponentTemplate<KSTrajControlLength>,
    public KSTrajExactControl,
    public KSTrajExactSpinControl,
    public KSTrajAdiabaticSpinControl,
    public KSTrajAdiabaticControl,
    public KSTrajExactTrappedControl,
    public KSTrajMagneticControl,
    public KSTrajElectricControl
{
  public:
    KSTrajControlLength();
    KSTrajControlLength(const KSTrajControlLength& aCopy);
    KSTrajControlLength* Clone() const override;
    ~KSTrajControlLength() override;

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

    void Calculate(const KSTrajExactTrappedParticle& aParticle, double& aValue) override;
    void Check(const KSTrajExactTrappedParticle& anInitialParticle, const KSTrajExactTrappedParticle& aFinalParticle,
               const KSTrajExactTrappedError& anError, bool& aFlag) override;

    void Calculate(const KSTrajMagneticParticle& aParticle, double& aValue) override;
    void Check(const KSTrajMagneticParticle& anInitialParticle, const KSTrajMagneticParticle& aFinalParticle,
               const KSTrajMagneticError& anError, bool& aFlag) override;

    void Calculate(const KSTrajElectricParticle& aParticle, double& aValue) override;
    void Check(const KSTrajElectricParticle& anInitialParticle, const KSTrajElectricParticle& aFinalParticle,
               const KSTrajElectricError& anError, bool& aFlag) override;

  public:
    void SetLength(const double& aLength);

  private:
    double fLength;
};

inline void KSTrajControlLength::SetLength(const double& aLength)
{
    fLength = aLength;
    return;
}

}  // namespace Kassiopeia

#endif
