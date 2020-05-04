#ifndef Kassiopeia_KSTrajInterpolatorFast_h_
#define Kassiopeia_KSTrajInterpolatorFast_h_

#include "KSComponentTemplate.h"
#include "KSTrajAdiabaticSpinTypes.h"
#include "KSTrajAdiabaticTypes.h"
#include "KSTrajElectricTypes.h"
#include "KSTrajExactSpinTypes.h"
#include "KSTrajExactTypes.h"
#include "KSTrajMagneticTypes.h"

namespace Kassiopeia
{

class KSTrajInterpolatorFast :
    public KSComponentTemplate<KSTrajInterpolatorFast>,
    public KSTrajExactInterpolator,
    public KSTrajExactSpinInterpolator,
    public KSTrajAdiabaticSpinInterpolator,
    public KSTrajAdiabaticInterpolator,
    public KSTrajElectricInterpolator,
    public KSTrajMagneticInterpolator
{
  public:
    KSTrajInterpolatorFast();
    KSTrajInterpolatorFast(const KSTrajInterpolatorFast& aCopy);
    KSTrajInterpolatorFast* Clone() const override;
    ~KSTrajInterpolatorFast() override;

  public:
    void Interpolate(double /*aTime*/, const KSTrajExactIntegrator& /*anIntegrator*/,
                     const KSTrajExactDifferentiator& /*aDifferentiator*/, const KSTrajExactParticle& anInitialParticle,
                     const KSTrajExactParticle& aFinalParticle, const double& aTimeStep,
                     KSTrajExactParticle& anIntermediateParticle) const override;

    void Interpolate(double /*aTime*/, const KSTrajExactSpinIntegrator& /*anIntegrator*/,
                     const KSTrajExactSpinDifferentiator& /*aDifferentiator*/,
                     const KSTrajExactSpinParticle& anInitialParticle, const KSTrajExactSpinParticle& aFinalParticle,
                     const double& aTimeStep, KSTrajExactSpinParticle& anIntermediateParticle) const override;

    void Interpolate(double /*aTime*/, const KSTrajAdiabaticSpinIntegrator& /*anIntegrator*/,
                     const KSTrajAdiabaticSpinDifferentiator& /*aDifferentiator*/,
                     const KSTrajAdiabaticSpinParticle& anInitialParticle,
                     const KSTrajAdiabaticSpinParticle& aFinalParticle, const double& aTimeStep,
                     KSTrajAdiabaticSpinParticle& anIntermediateParticle) const override;

    void Interpolate(double /*aTime*/, const KSTrajAdiabaticIntegrator& /*anIntegrator*/,
                     const KSTrajAdiabaticDifferentiator& /*aDifferentiator*/, const KSTrajAdiabaticParticle& anInitial,
                     const KSTrajAdiabaticParticle& aFinal, const double& aValue,
                     KSTrajAdiabaticParticle& anIntermediate) const override;

    void Interpolate(double /*aTime*/, const KSTrajMagneticIntegrator& /*anIntegrator*/,
                     const KSTrajMagneticDifferentiator& /*aDifferentiator*/, const KSTrajMagneticParticle& anInitial,
                     const KSTrajMagneticParticle& aFinal, const double& aValue,
                     KSTrajMagneticParticle& anIntermediate) const override;

    void Interpolate(double /*aTime*/, const KSTrajElectricIntegrator& /*anIntegrator*/,
                     const KSTrajElectricDifferentiator& /*aDifferentiator*/, const KSTrajElectricParticle& anInitial,
                     const KSTrajElectricParticle& aFinal, const double& aValue,
                     KSTrajElectricParticle& anIntermediate) const override;

    //we replace the generic linear approximation with a single line segment approximation
    void GetPiecewiseLinearApproximation(double /*aTolerance*/, unsigned int /*nMaxSegments*/, double /*anInitialTime*/,
                                         double /*aFinalTime*/, const KSTrajExactIntegrator& /*anIntegrator*/,
                                         const KSTrajExactDifferentiator& /*aDifferentiator*/,
                                         const KSTrajExactParticle& anInitialValue,
                                         const KSTrajExactParticle& aFinalValue,
                                         std::vector<KSTrajExactParticle>* interpolatedValues) const override;

    void GetPiecewiseLinearApproximation(double /*aTolerance*/, unsigned int /*nMaxSegments*/, double /*anInitialTime*/,
                                         double /*aFinalTime*/, const KSTrajExactSpinIntegrator& /*anIntegrator*/,
                                         const KSTrajExactSpinDifferentiator& /*aDifferentiator*/,
                                         const KSTrajExactSpinParticle& anInitialValue,
                                         const KSTrajExactSpinParticle& aFinalValue,
                                         std::vector<KSTrajExactSpinParticle>* interpolatedValues) const override;

    void GetPiecewiseLinearApproximation(double /*aTolerance*/, unsigned int /*nMaxSegments*/, double /*anInitialTime*/,
                                         double /*aFinalTime*/, const KSTrajAdiabaticSpinIntegrator& /*anIntegrator*/,
                                         const KSTrajAdiabaticSpinDifferentiator& /*aDifferentiator*/,
                                         const KSTrajAdiabaticSpinParticle& anInitialValue,
                                         const KSTrajAdiabaticSpinParticle& aFinalValue,
                                         std::vector<KSTrajAdiabaticSpinParticle>* interpolatedValues) const override;

    void GetPiecewiseLinearApproximation(double /*aTolerance*/, unsigned int /*nMaxSegments*/, double /*anInitialTime*/,
                                         double /*aFinalTime*/, const KSTrajAdiabaticIntegrator& /*anIntegrator*/,
                                         const KSTrajAdiabaticDifferentiator& /*aDifferentiator*/,
                                         const KSTrajAdiabaticParticle& anInitialValue,
                                         const KSTrajAdiabaticParticle& aFinalValue,
                                         std::vector<KSTrajAdiabaticParticle>* interpolatedValues) const override;

    void GetPiecewiseLinearApproximation(double /*aTolerance*/, unsigned int /*nMaxSegments*/, double /*anInitialTime*/,
                                         double /*aFinalTime*/, const KSTrajMagneticIntegrator& /*anIntegrator*/,
                                         const KSTrajMagneticDifferentiator& /*aDifferentiator*/,
                                         const KSTrajMagneticParticle& anInitialValue,
                                         const KSTrajMagneticParticle& aFinalValue,
                                         std::vector<KSTrajMagneticParticle>* interpolatedValues) const override;

    void GetPiecewiseLinearApproximation(double /*aTolerance*/, unsigned int /*nMaxSegments*/, double /*anInitialTime*/,
                                         double /*aFinalTime*/, const KSTrajElectricIntegrator& /*anIntegrator*/,
                                         const KSTrajElectricDifferentiator& /*aDifferentiator*/,
                                         const KSTrajElectricParticle& anInitialValue,
                                         const KSTrajElectricParticle& aFinalValue,
                                         std::vector<KSTrajElectricParticle>* interpolatedValues) const override;
};

}  // namespace Kassiopeia

#endif
