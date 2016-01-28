#ifndef Kassiopeia_KSTrajInterpolatorHermite_h_
#define Kassiopeia_KSTrajInterpolatorHermite_h_

#include "KSComponentTemplate.h"

#include "KSTrajExactTypes.h"
#include "KSTrajAdiabaticTypes.h"
#include "KSTrajMagneticTypes.h"
#include "KSTrajElectricTypes.h"
#include "KSTrajInterpolatorFast.h"

namespace Kassiopeia
{

    class KSTrajInterpolatorHermite :
        public KSComponentTemplate< KSTrajInterpolatorHermite >,
        public KSTrajExactInterpolator,
        public KSTrajAdiabaticInterpolator,
        public KSTrajMagneticInterpolator,
        public KSTrajElectricInterpolator
    {
        public:
            KSTrajInterpolatorHermite();
            KSTrajInterpolatorHermite( const KSTrajInterpolatorHermite& aCopy );
            KSTrajInterpolatorHermite* Clone() const;
            virtual ~KSTrajInterpolatorHermite();

        public:

            virtual void Interpolate(double aTime, const KSTrajExactIntegrator& anIntegrator, const KSTrajExactDifferentiator& aDifferentiator, const KSTrajExactParticle& anInitialParticle, const KSTrajExactParticle& aFinalParticle, const double& aTimeStep, KSTrajExactParticle& anIntermediateParticle ) const;

            virtual void Interpolate(double aTime, const KSTrajAdiabaticIntegrator& anIntegrator, const KSTrajAdiabaticDifferentiator& aDifferentiator, const KSTrajAdiabaticParticle& anInitial, const KSTrajAdiabaticParticle& aFinal, const double& aValue, KSTrajAdiabaticParticle& anIntermediate ) const;

            virtual void Interpolate(double aTime, const KSTrajMagneticIntegrator& anIntegrator, const KSTrajMagneticDifferentiator& aDifferentiator, const KSTrajMagneticParticle& anInitial, const KSTrajMagneticParticle& aFinal, const double& aValue, KSTrajMagneticParticle& anIntermediate ) const;

            virtual void Interpolate(double aTime, const KSTrajElectricIntegrator& anIntegrator, const KSTrajElectricDifferentiator& aDifferentiator, const KSTrajElectricParticle& anInitial, const KSTrajElectricParticle& aFinal, const double& aValue, KSTrajElectricParticle& anIntermediate ) const;


            //we use the generic linear approximation these distance metrics
            virtual double DistanceMetric(const KSTrajExactParticle& valueA, const KSTrajExactParticle& valueB) const;
            virtual double DistanceMetric(const KSTrajAdiabaticParticle& valueA, const KSTrajAdiabaticParticle& valueB) const;
            virtual double DistanceMetric(const KSTrajMagneticParticle& valueA, const KSTrajMagneticParticle& valueB) const;
            virtual double DistanceMetric(const KSTrajElectricParticle& valueA, const KSTrajElectricParticle& valueB) const;

        public:
            //evaluate cubic hermite basis functions on [0,1]
            static void CubicHermite(double t, double& h30, double& h31, double& h32, double& h33);

            //evaluate quintic hermite basis functions on [0,1]
            static void QuinticHermite(double t, double& h50, double& h51, double& h52, double& h53, double& h54, double& h55);

            //fast linear interpolator, used for extrapolation beyond current time step
            KSTrajInterpolatorFast fFastInterpolator;

    };

}

#endif
