#ifndef Kassiopeia_KSTrajInterpolatorContinuousRungeKutta_h_
#define Kassiopeia_KSTrajInterpolatorContinuousRungeKutta_h_

#include "KSComponentTemplate.h"

#include "KSTrajExactTypes.h"
#include "KSTrajExactSpinTypes.h"
#include "KSTrajAdiabaticSpinTypes.h"
#include "KSTrajAdiabaticTypes.h"
#include "KSTrajMagneticTypes.h"
#include "KSTrajElectricTypes.h"
#include "KSTrajInterpolatorHermite.h"

namespace Kassiopeia
{

    class KSTrajInterpolatorContinuousRungeKutta :
        public KSComponentTemplate< KSTrajInterpolatorContinuousRungeKutta >,
        public KSTrajExactInterpolator,
        public KSTrajExactSpinInterpolator,
        public KSTrajAdiabaticSpinInterpolator,
        public KSTrajAdiabaticInterpolator,
        public KSTrajMagneticInterpolator,
        public KSTrajElectricInterpolator
    {
        public:
            KSTrajInterpolatorContinuousRungeKutta();
            KSTrajInterpolatorContinuousRungeKutta( const KSTrajInterpolatorContinuousRungeKutta& aCopy );
            KSTrajInterpolatorContinuousRungeKutta* Clone() const;
            virtual ~KSTrajInterpolatorContinuousRungeKutta();

        public:

            virtual void Interpolate(double aTime, const KSTrajExactIntegrator& anIntegrator, const KSTrajExactDifferentiator& aDifferentiator, const KSTrajExactParticle& anInitialParticle, const KSTrajExactParticle& aFinalParticle, const double& aTimeStep, KSTrajExactParticle& anIntermediateParticle ) const;

            virtual void Interpolate(double aTime, const KSTrajExactSpinIntegrator& anIntegrator, const KSTrajExactSpinDifferentiator& aDifferentiator, const KSTrajExactSpinParticle& anInitialParticle, const KSTrajExactSpinParticle& aFinalParticle, const double& aTimeStep, KSTrajExactSpinParticle& anIntermediateParticle ) const;

            virtual void Interpolate(double aTime, const KSTrajAdiabaticSpinIntegrator& anIntegrator, const KSTrajAdiabaticSpinDifferentiator& aDifferentiator, const KSTrajAdiabaticSpinParticle& anInitialParticle, const KSTrajAdiabaticSpinParticle& aFinalParticle, const double& aTimeStep, KSTrajAdiabaticSpinParticle& anIntermediateParticle ) const;

            virtual void Interpolate(double aTime, const KSTrajAdiabaticIntegrator& anIntegrator, const KSTrajAdiabaticDifferentiator& aDifferentiator, const KSTrajAdiabaticParticle& anInitial, const KSTrajAdiabaticParticle& aFinal, const double& aValue, KSTrajAdiabaticParticle& anIntermediate ) const;

            virtual void Interpolate(double aTime, const KSTrajMagneticIntegrator& anIntegrator, const KSTrajMagneticDifferentiator& aDifferentiator, const KSTrajMagneticParticle& anInitial, const KSTrajMagneticParticle& aFinal, const double& aValue, KSTrajMagneticParticle& anIntermediate ) const;

            virtual void Interpolate(double aTime, const KSTrajElectricIntegrator& anIntegrator, const KSTrajElectricDifferentiator& aDifferentiator, const KSTrajElectricParticle& anInitial, const KSTrajElectricParticle& aFinal, const double& aValue, KSTrajElectricParticle& anIntermediate ) const;

            //we use the generic linear approximation these distance metrics
            virtual double DistanceMetric(const KSTrajExactParticle& valueA, const KSTrajExactParticle& valueB) const;
            virtual double DistanceMetric(const KSTrajExactSpinParticle& valueA, const KSTrajExactSpinParticle& valueB) const;
            virtual double DistanceMetric(const KSTrajAdiabaticSpinParticle& valueA, const KSTrajAdiabaticSpinParticle& valueB) const;
            virtual double DistanceMetric(const KSTrajAdiabaticParticle& valueA, const KSTrajAdiabaticParticle& valueB) const;
            virtual double DistanceMetric(const KSTrajMagneticParticle& valueA, const KSTrajMagneticParticle& valueB) const;
            virtual double DistanceMetric(const KSTrajElectricParticle& valueA, const KSTrajElectricParticle& valueB) const;


        public:

            //hermite interpolator, default for when the Runge-Kutta integrator
            //does not support a continuous extension
            KSTrajInterpolatorHermite fHermiteInterpolator;

    };

}

#endif
