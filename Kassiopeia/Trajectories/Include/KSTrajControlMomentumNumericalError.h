#ifndef Kassiopeia_KSTrajControlMomentumNumericalError_h_
#define Kassiopeia_KSTrajControlMomentumNumericalError_h_

#include "KSComponentTemplate.h"
#include "KSTrajAdiabaticSpinTypes.h"
#include "KSTrajAdiabaticTypes.h"
#include "KSTrajExactSpinTypes.h"
#include "KSTrajExactTypes.h"

namespace Kassiopeia
{

class KSTrajControlMomentumNumericalError :
    public KSComponentTemplate<KSTrajControlMomentumNumericalError>,
    public KSTrajExactControl,
    public KSTrajExactSpinControl,
    public KSTrajAdiabaticSpinControl,
    public KSTrajAdiabaticControl
{
  public:
    KSTrajControlMomentumNumericalError();
    KSTrajControlMomentumNumericalError(const KSTrajControlMomentumNumericalError& aCopy);
    KSTrajControlMomentumNumericalError* Clone() const override;
    ~KSTrajControlMomentumNumericalError() override;

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


  protected:
    virtual void ActivateObject();

  public:
    void SetAbsoluteMomentumError(double error)
    {
        fAbsoluteError = error;
    };
    void SetSafetyFactor(double safety)
    {
        fSafetyFactor = safety;
    };
    void SetSolverOrder(double order)
    {
        fSolverOrder = order;
    };

  private:
    bool UpdateTimeStep(double error);

    static double fEpsilon;

    double fAbsoluteError;  //max allowable error on momentum magnitude per step
    double fSafetyFactor;   //safety factor for increasing/decreasing step size
    double fSolverOrder;    //order of the associated runge-kutta stepper

    double fTimeStep;
    bool fFirstStep;
};

}  // namespace Kassiopeia

#endif
