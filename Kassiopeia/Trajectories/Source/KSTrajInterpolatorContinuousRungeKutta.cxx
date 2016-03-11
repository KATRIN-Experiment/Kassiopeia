#include "KSTrajInterpolatorContinuousRungeKutta.h"


#include "KSTrajectoriesMessage.h"

namespace Kassiopeia
{

    KSTrajInterpolatorContinuousRungeKutta::KSTrajInterpolatorContinuousRungeKutta()
    {
    }
    KSTrajInterpolatorContinuousRungeKutta::KSTrajInterpolatorContinuousRungeKutta( const KSTrajInterpolatorContinuousRungeKutta& ):
        KSComponent()
    {
    }
    KSTrajInterpolatorContinuousRungeKutta* KSTrajInterpolatorContinuousRungeKutta::Clone() const
    {
        return new KSTrajInterpolatorContinuousRungeKutta( *this );
    }
    KSTrajInterpolatorContinuousRungeKutta::~KSTrajInterpolatorContinuousRungeKutta()
    {
    }

    void KSTrajInterpolatorContinuousRungeKutta::Interpolate(double aTime,
                                                const KSTrajExactIntegrator& anIntegrator,
                                                const KSTrajExactDifferentiator& aDifferentiator,
                                                const KSTrajExactParticle& anInitialParticle,
                                                const KSTrajExactParticle& aFinalParticle,
                                                const double& aTimeStep,
                                                KSTrajExactParticle& anIntermediateParticle ) const
    {
        //exact particle data:
        //0 is time
        //1 is length
        //2 is x component of position
        //3 is y component of position
        //4 is z component of position
        //5 is x component of momentum
        //6 is y component of momentum
        //7 is z component of momentum

        //compute the time step data
        double tInitialTime = anInitialParticle.GetTime();
        double tFinalTime = aFinalParticle.GetTime();
        double tDeltaTime = tFinalTime - tInitialTime;
        double tFraction = aTimeStep / tDeltaTime;

        bool validRange = false;
        if( tFraction >= 0.0 && tFraction <= 1.0){validRange = true;};

        if(validRange && anIntegrator.HasDenseOutput())
        {
            //the integrator supports interpolation, compute the needed value
            anIntegrator.Interpolate(tFraction, anIntermediateParticle);
            return;
        }
        else
        {
            //no support from this integrator, default to hermite interpolant
            trajmsg_debug( "time step out of range or current ode integrator not equiped with dense output, defaulting to hermite interpolator." << eom );

            fHermiteInterpolator.Interpolate(aTime, anIntegrator, aDifferentiator, anInitialParticle, aFinalParticle, aTimeStep, anIntermediateParticle);
            return;
        }
    }

    void KSTrajInterpolatorContinuousRungeKutta::Interpolate(double aTime, const KSTrajAdiabaticIntegrator& anIntegrator, const KSTrajAdiabaticDifferentiator& aDifferentiator, const KSTrajAdiabaticParticle& anInitialParticle, const KSTrajAdiabaticParticle& aFinalParticle, const double& aTimeStep, KSTrajAdiabaticParticle& anIntermediateParticle ) const
    {
        //we only do cubic interpolation on the adiabatic particle

        //adiabatic particle data:
        //0 is time
        //1 is length
        //2 is x component of guiding center
        //3 is y component of guiding center
        //4 is z component of guiding center
        //5 is longitudinal momentum
        //6 is transverse momentum
        //7 is phase

        //compute the time step data
        double tInitialTime = anInitialParticle.GetTime();
        double tFinalTime = aFinalParticle.GetTime();
        double tDeltaTime = tFinalTime - tInitialTime;
        double tFraction = aTimeStep / tDeltaTime;

        bool validRange = false;
        if( tFraction >= 0.0 && tFraction <= 1.0){validRange = true;};

        if(validRange && anIntegrator.HasDenseOutput())
        {
            if( tFraction < 1e-13 )
            {
                anIntermediateParticle = anInitialParticle;
            }
            else if( std::fabs(1.0 - tFraction) < 1e-13)
            {
                anIntermediateParticle = aFinalParticle;
            }
            else
            {
                //the integrator supports interpolation, compute the needed value
                anIntegrator.Interpolate(tFraction, anIntermediateParticle);
            }

            //interpolate alpha and beta linearly
            //(leaving this unchanged from fast interpolator, may need to improve this)
            KThreeVector tInitialAlpha = anInitialParticle.GetAlpha().Unit();
            KThreeVector tFinalAlpha = aFinalParticle.GetAlpha().Unit();
            KThreeVector tRotationVectorAlpha = tInitialAlpha.Cross( tFinalAlpha );
            double tRotationAngleAlpha = tFraction * asin( tRotationVectorAlpha.Magnitude() );
            KThreeVector tRotationAxisAlpha = tRotationVectorAlpha.Unit();
            KThreeVector tInterpolatedAlpha = cos( tRotationAngleAlpha ) * tInitialAlpha + sin( tRotationAngleAlpha ) * tRotationAxisAlpha.Cross( tInitialAlpha );

            KThreeVector tInitialBeta = anInitialParticle.GetBeta().Unit();
            KThreeVector tFinalBeta = aFinalParticle.GetBeta().Unit();
            KThreeVector tRotationVectorBeta = tInitialBeta.Cross( tFinalBeta );
            double tRotationAngleBeta = tFraction * asin( tRotationVectorBeta.Magnitude() );
            KThreeVector tRotationAxisBeta = tRotationVectorBeta.Unit();
            KThreeVector tInterpolatedBeta = cos( tRotationAngleBeta ) * tInitialBeta + sin( tRotationAngleBeta ) * tRotationAxisBeta.Cross( tInitialBeta );

            anIntermediateParticle.SetAlpha( tInterpolatedAlpha );
            anIntermediateParticle.SetBeta( tInterpolatedBeta );

            return;
        }
        else
        {
            trajmsg_debug( "time step out of range or current ode integrator not equiped with dense output, defaulting to hermite interpolator." << eom );

            //no support from this integrator, default to hermite interpolant
            fHermiteInterpolator.Interpolate(aTime, anIntegrator, aDifferentiator, anInitialParticle, aFinalParticle, aTimeStep, anIntermediateParticle);
            return;
        }
    }

    void KSTrajInterpolatorContinuousRungeKutta::Interpolate(double aTime, const KSTrajMagneticIntegrator& anIntegrator, const KSTrajMagneticDifferentiator& aDifferentiator, const KSTrajMagneticParticle& anInitialParticle, const KSTrajMagneticParticle& aFinalParticle, const double& aTimeStep, KSTrajMagneticParticle& anIntermediateParticle ) const
    {
        //compute the time step data
        double tInitialTime = anInitialParticle.GetTime();
        double tFinalTime = aFinalParticle.GetTime();
        double tDeltaTime = tFinalTime - tInitialTime;
        double tFraction = aTimeStep / tDeltaTime;

        bool validRange = false;
        if( tFraction >= 0.0 && tFraction <= 1.0){validRange = true;};

        if(validRange && anIntegrator.HasDenseOutput())
        {
            //the integrator supports interpolation, compute the needed value
            anIntegrator.Interpolate(tFraction, anIntermediateParticle);
            return;
        }
        else
        {
            //no support from this integrator, default to hermite interpolant
            trajmsg_debug( "time step out of range or current ode integrator not equiped with dense output, defaulting to hermite interpolator." << eom );

            fHermiteInterpolator.Interpolate(aTime, anIntegrator, aDifferentiator, anInitialParticle, aFinalParticle, aTimeStep, anIntermediateParticle);
            return;
        }
    }

    void KSTrajInterpolatorContinuousRungeKutta::Interpolate(double aTime, const KSTrajElectricIntegrator& anIntegrator, const KSTrajElectricDifferentiator& aDifferentiator, const KSTrajElectricParticle& anInitialParticle, const KSTrajElectricParticle& aFinalParticle, const double& aTimeStep, KSTrajElectricParticle& anIntermediateParticle ) const
    {
        //compute the time step data
        double tInitialTime = anInitialParticle.GetTime();
        double tFinalTime = aFinalParticle.GetTime();
        double tDeltaTime = tFinalTime - tInitialTime;
        double tFraction = aTimeStep / tDeltaTime;

        bool validRange = false;
        if( tFraction >= 0.0 && tFraction <= 1.0){validRange = true;};

        if(validRange && anIntegrator.HasDenseOutput())
        {
            //the integrator supports interpolation, compute the needed value
            anIntegrator.Interpolate(tFraction, anIntermediateParticle);
            return;
        }
        else
        {
            //no support from this integrator, default to hermite interpolant
            trajmsg_debug( "time step out of range or current ode integrator not equiped with dense output, defaulting to hermite interpolator." << eom );

            fHermiteInterpolator.Interpolate(aTime, anIntegrator, aDifferentiator, anInitialParticle, aFinalParticle, aTimeStep, anIntermediateParticle);
            return;
        }
    }

    double
    KSTrajInterpolatorContinuousRungeKutta::DistanceMetric(const KSTrajExactParticle& valueA, const KSTrajExactParticle& valueB) const
    {
        KThreeVector a = valueA.GetPosition();
        KThreeVector b = valueB.GetPosition();
        return (a-b).Magnitude();
    }

    double
    KSTrajInterpolatorContinuousRungeKutta::DistanceMetric(const KSTrajAdiabaticParticle& valueA, const KSTrajAdiabaticParticle& valueB) const
    {
        KThreeVector a = valueA.GetGuidingCenter();
        KThreeVector b = valueB.GetGuidingCenter();
        return (a-b).Magnitude();
    }

    double
    KSTrajInterpolatorContinuousRungeKutta::DistanceMetric(const KSTrajMagneticParticle& valueA, const KSTrajMagneticParticle& valueB) const
    {
        KThreeVector a = valueA.GetPosition();
        KThreeVector b = valueB.GetPosition();
        return (a-b).Magnitude();
    }

    double
    KSTrajInterpolatorContinuousRungeKutta::DistanceMetric(const KSTrajElectricParticle& valueA, const KSTrajElectricParticle& valueB) const
    {
        KThreeVector a = valueA.GetPosition();
        KThreeVector b = valueB.GetPosition();
        return (a-b).Magnitude();
    }



}
