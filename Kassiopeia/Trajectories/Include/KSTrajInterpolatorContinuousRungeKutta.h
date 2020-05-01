#ifndef Kassiopeia_KSTrajInterpolatorContinuousRungeKutta_h_
#define Kassiopeia_KSTrajInterpolatorContinuousRungeKutta_h_

#include "KSComponentTemplate.h"
#include "KSTrajAdiabaticSpinTypes.h"
#include "KSTrajAdiabaticTypes.h"
#include "KSTrajElectricTypes.h"
#include "KSTrajExactSpinTypes.h"
#include "KSTrajExactTypes.h"
#include "KSTrajInterpolatorHermite.h"
#include "KSTrajMagneticTypes.h"

namespace Kassiopeia
{

class KSTrajInterpolatorContinuousRungeKutta :
    public KSComponentTemplate<KSTrajInterpolatorContinuousRungeKutta>,
    public KSTrajExactInterpolator,
    public KSTrajExactSpinInterpolator,
    public KSTrajAdiabaticSpinInterpolator,
    public KSTrajAdiabaticInterpolator,
    public KSTrajMagneticInterpolator,
    public KSTrajElectricInterpolator
{
  public:
    KSTrajInterpolatorContinuousRungeKutta();
    KSTrajInterpolatorContinuousRungeKutta(const KSTrajInterpolatorContinuousRungeKutta& aCopy);
    KSTrajInterpolatorContinuousRungeKutta* Clone() const override;
    ~KSTrajInterpolatorContinuousRungeKutta() override;

  public:
    void Interpolate(double aTime, const KSTrajExactIntegrator& anIntegrator,
                     const KSTrajExactDifferentiator& aDifferentiator, const KSTrajExactParticle& anInitialParticle,
                     const KSTrajExactParticle& aFinalParticle, const double& aTimeStep,
                     KSTrajExactParticle& anIntermediateParticle) const override;

    void Interpolate(double aTime, const KSTrajExactSpinIntegrator& anIntegrator,
                     const KSTrajExactSpinDifferentiator& aDifferentiator,
                     const KSTrajExactSpinParticle& anInitialParticle, const KSTrajExactSpinParticle& aFinalParticle,
                     const double& aTimeStep, KSTrajExactSpinParticle& anIntermediateParticle) const override;

    void Interpolate(double aTime, const KSTrajAdiabaticSpinIntegrator& anIntegrator,
                     const KSTrajAdiabaticSpinDifferentiator& aDifferentiator,
                     const KSTrajAdiabaticSpinParticle& anInitialParticle,
                     const KSTrajAdiabaticSpinParticle& aFinalParticle, const double& aTimeStep,
                     KSTrajAdiabaticSpinParticle& anIntermediateParticle) const override;

    void Interpolate(double aTime, const KSTrajAdiabaticIntegrator& anIntegrator,
                     const KSTrajAdiabaticDifferentiator& aDifferentiator, const KSTrajAdiabaticParticle& anInitial,
                     const KSTrajAdiabaticParticle& aFinal, const double& aValue,
                     KSTrajAdiabaticParticle& anIntermediate) const override;

    void Interpolate(double aTime, const KSTrajMagneticIntegrator& anIntegrator,
                     const KSTrajMagneticDifferentiator& aDifferentiator, const KSTrajMagneticParticle& anInitial,
                     const KSTrajMagneticParticle& aFinal, const double& aValue,
                     KSTrajMagneticParticle& anIntermediate) const override;

    void Interpolate(double aTime, const KSTrajElectricIntegrator& anIntegrator,
                     const KSTrajElectricDifferentiator& aDifferentiator, const KSTrajElectricParticle& anInitial,
                     const KSTrajElectricParticle& aFinal, const double& aValue,
                     KSTrajElectricParticle& anIntermediate) const override;

    //we use the generic linear approximation these distance metrics
    double DistanceMetric(const KSTrajExactParticle& valueA, const KSTrajExactParticle& valueB) const override;
    double DistanceMetric(const KSTrajExactSpinParticle& valueA, const KSTrajExactSpinParticle& valueB) const override;
    double DistanceMetric(const KSTrajAdiabaticSpinParticle& valueA,
                          const KSTrajAdiabaticSpinParticle& valueB) const override;
    double DistanceMetric(const KSTrajAdiabaticParticle& valueA, const KSTrajAdiabaticParticle& valueB) const override;
    double DistanceMetric(const KSTrajMagneticParticle& valueA, const KSTrajMagneticParticle& valueB) const override;
    double DistanceMetric(const KSTrajElectricParticle& valueA, const KSTrajElectricParticle& valueB) const override;


  public:
    //hermite interpolator, default for when the Runge-Kutta integrator
    //does not support a continuous extension
    KSTrajInterpolatorHermite fHermiteInterpolator;
};

}  // namespace Kassiopeia

#endif
