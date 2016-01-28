#include "KSTrajInterpolatorHermite.h"
#include "KSTrajectoriesMessage.h"

#include <list>

namespace Kassiopeia
{

    KSTrajInterpolatorHermite::KSTrajInterpolatorHermite()
    {
    }
    KSTrajInterpolatorHermite::KSTrajInterpolatorHermite( const KSTrajInterpolatorHermite& ):
        KSComponent()
    {
    }
    KSTrajInterpolatorHermite* KSTrajInterpolatorHermite::Clone() const
    {
        return new KSTrajInterpolatorHermite( *this );
    }
    KSTrajInterpolatorHermite::~KSTrajInterpolatorHermite()
    {
    }

    void KSTrajInterpolatorHermite::Interpolate(double aTime,
                                                const KSTrajExactIntegrator& anIntegrator,
                                                const KSTrajExactDifferentiator& aDifferentiator,
                                                const KSTrajExactParticle& anInitialParticle,
                                                const KSTrajExactParticle& aFinalParticle,
                                                const double& aTimeStep,
                                                KSTrajExactParticle& anIntermediateParticle ) const
    {
        //cubic interpolation on all variables, except postion, which is quintic

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
        double tInterpolatedTime = tInitialTime + aTimeStep;

        //time step must be within the range specified by the last step
        if( tFraction >= 0.0 && tFraction <= 1.0)
        {
            //compute the cubic hermite polynomials
            double h30, h31, h32, h33;
            CubicHermite(tFraction, h30, h31, h32, h33);

            //retrieve the first derivative evaluation from the integrator
            KSTrajExactDerivative initialDerivative;
            initialDerivative = 0.0;;
            bool isValid = anIntegrator.GetInitialDerivative(initialDerivative );
            if(!isValid)
            {
                aDifferentiator.Differentiate(anInitialParticle.GetTime(), anInitialParticle, initialDerivative);
            }

            //retrieve the final derivative evaluation from the integrator
            KSTrajExactDerivative finalDerivative;
            finalDerivative = 0.0;
            isValid = anIntegrator.GetFinalDerivative(finalDerivative );
            if(!isValid)
            {
                aDifferentiator.Differentiate(aFinalParticle.GetTime(), aFinalParticle, finalDerivative);
            }

            //compute the cubic interpolation for all variables in the particle array
            anIntermediateParticle = h30*anInitialParticle + h31*tDeltaTime*initialDerivative + h32*tDeltaTime*finalDerivative + h33*aFinalParticle;
            anIntermediateParticle[0] = tInterpolatedTime; //explicitly set the time variable

            return;
        }
        else
        {
            //outside of step range, use linear interpolator to extrapolate
            trajmsg_debug( "time step out of range defaulting to linear interpolator." << eom );

            fFastInterpolator.Interpolate(aTime, anIntegrator, aDifferentiator, anInitialParticle, aFinalParticle, aTimeStep, anIntermediateParticle);
        }
    }

    void KSTrajInterpolatorHermite::Interpolate(double aTime,
                                                const KSTrajAdiabaticIntegrator& anIntegrator,
                                                const KSTrajAdiabaticDifferentiator& aDifferentiator,
                                                const KSTrajAdiabaticParticle& anInitialParticle,
                                                const KSTrajAdiabaticParticle& aFinalParticle,
                                                const double& aTimeStep,
                                                KSTrajAdiabaticParticle& anIntermediateParticle ) const
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
        double tFraction = aTimeStep / (tFinalTime - tInitialTime);
        double tInterpolatedTime = tInitialTime + aTimeStep;

        if( tFraction >= 0.0 && tFraction <= 1.0)
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
                //compute the cubic hermite polynomials
                double h30, h31, h32, h33;
                CubicHermite(tFraction, h30, h31, h32, h33);

                //retrieve the first derivative evaluation from the integrator
                KSTrajAdiabaticDerivative initialDerivative;
                initialDerivative = 0.0;
                bool isValid = anIntegrator.GetInitialDerivative(initialDerivative);
                if(!isValid)
                {
                    aDifferentiator.Differentiate(anInitialParticle.GetTime(), anInitialParticle, initialDerivative);
                }

                //retrieve the final derivative evaluation from the integrator
                KSTrajAdiabaticDerivative finalDerivative;
                finalDerivative = 0.0;
                isValid = anIntegrator.GetFinalDerivative(finalDerivative);
                if(!isValid)
                {
                    aDifferentiator.Differentiate(aFinalParticle.GetTime(), aFinalParticle, finalDerivative);
                }

                //compute the cubic interpolation for all variables in the particle array
                anIntermediateParticle = h30*anInitialParticle + h31*tDeltaTime*initialDerivative + h32*tDeltaTime*finalDerivative + h33*aFinalParticle;
                anIntermediateParticle[0] = tInterpolatedTime; //explicitly set the time variable
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
        }
        else
        {
            //outside of step range, use linear interpolator to extrapolate
            trajmsg_debug( "time step out of range defaulting to linear interpolator." << eom );

            fFastInterpolator.Interpolate(aTime, anIntegrator, aDifferentiator, anInitialParticle, aFinalParticle, aTimeStep, anIntermediateParticle);
        }
    }

    void KSTrajInterpolatorHermite::Interpolate(double aTime,
                                                const KSTrajMagneticIntegrator& anIntegrator,
                                                const KSTrajMagneticDifferentiator& aDifferentiator,
                                                const KSTrajMagneticParticle& anInitialParticle,
                                                const KSTrajMagneticParticle& aFinalParticle,
                                                const double& aTimeStep,
                                                KSTrajMagneticParticle& anIntermediateParticle ) const
    {
        //only do cubic interpolation for the magnetic particle

        //compute the time step data
        double tInitialTime = anInitialParticle.GetTime();
        double tFinalTime = aFinalParticle.GetTime();
        double tDeltaTime = tFinalTime - tInitialTime;
        double tFraction = aTimeStep / (tFinalTime - tInitialTime);
        double tInterpolatedTime = tInitialTime + aTimeStep;

        if( tFraction >= 0.0 && tFraction <= 1.0)
        {
            //compute the cubic hermite polynomials
            double h30, h31, h32, h33;
            CubicHermite(tFraction, h30, h31, h32, h33);

            //retrieve the first derivative evaluation from the integrator
            KSTrajMagneticDerivative initialDerivative;
            initialDerivative = 0.0;
            bool isValid = anIntegrator.GetInitialDerivative(initialDerivative);
            if(!isValid)
            {
                aDifferentiator.Differentiate(anInitialParticle.GetTime(), anInitialParticle, initialDerivative);
            }

            //retrieve the final derivative evaluation from the integrator
            KSTrajMagneticDerivative finalDerivative;
            finalDerivative = 0.0;
            isValid = anIntegrator.GetFinalDerivative(finalDerivative);
            if(!isValid)
            {
                aDifferentiator.Differentiate(aFinalParticle.GetTime(), aFinalParticle, finalDerivative);
            }

            //compute the cubic interpolation for all variables in the particle array
            anIntermediateParticle = h30*anInitialParticle + h31*tDeltaTime*initialDerivative + h32*tDeltaTime*finalDerivative + h33*aFinalParticle;
            anIntermediateParticle[0] = tInterpolatedTime; //explicitly set the time variable
        }
        else
        {
            //outside of step range, use linear interpolator to extrapolate
            trajmsg_debug( "time step out of range defaulting to linear interpolator." << eom );

            fFastInterpolator.Interpolate(aTime, anIntegrator, aDifferentiator, anInitialParticle, aFinalParticle, aTimeStep, anIntermediateParticle);
        }
    }


    void KSTrajInterpolatorHermite::Interpolate(double aTime,
                                                const KSTrajElectricIntegrator& anIntegrator,
                                                const KSTrajElectricDifferentiator& aDifferentiator,
                                                const KSTrajElectricParticle& anInitialParticle,
                                                const KSTrajElectricParticle& aFinalParticle,
                                                const double& aTimeStep,
                                                KSTrajElectricParticle& anIntermediateParticle ) const
    {
        //only do cubic interpolation for the magnetic particle

        //compute the time step data
        double tInitialTime = anInitialParticle.GetTime();
        double tFinalTime = aFinalParticle.GetTime();
        double tDeltaTime = tFinalTime - tInitialTime;
        double tFraction = aTimeStep / (tFinalTime - tInitialTime);
        double tInterpolatedTime = tInitialTime + aTimeStep;

        if( tFraction >= 0.0 && tFraction <= 1.0)
        {
            //compute the cubic hermite polynomials
            double h30, h31, h32, h33;
            CubicHermite(tFraction, h30, h31, h32, h33);

            //retrieve the first derivative evaluation from the integrator
            KSTrajElectricDerivative initialDerivative;
            initialDerivative = 0.0;
            bool isValid = anIntegrator.GetInitialDerivative(initialDerivative);
            if(!isValid)
            {
                aDifferentiator.Differentiate(anInitialParticle.GetTime(), anInitialParticle, initialDerivative);
            }

            //retrieve the final derivative evaluation from the integrator
            KSTrajElectricDerivative finalDerivative;
            finalDerivative = 0.0;
            isValid = anIntegrator.GetFinalDerivative(finalDerivative);
            if(!isValid)
            {
                aDifferentiator.Differentiate(aFinalParticle.GetTime(), aFinalParticle, finalDerivative);
            }

            //compute the cubic interpolation for all variables in the particle array
            anIntermediateParticle = h30*anInitialParticle + h31*tDeltaTime*initialDerivative + h32*tDeltaTime*finalDerivative + h33*aFinalParticle;
            anIntermediateParticle[0] = tInterpolatedTime; //explicitly set the time variable
        }
        else
        {
            //outside of step range, use linear interpolator to extrapolate
            trajmsg_debug( "time step out of range defaulting to linear interpolator." << eom );

            fFastInterpolator.Interpolate(aTime, anIntegrator, aDifferentiator, anInitialParticle, aFinalParticle, aTimeStep, anIntermediateParticle);
        }
    }


    double
    KSTrajInterpolatorHermite::DistanceMetric(const KSTrajExactParticle& valueA, const KSTrajExactParticle& valueB) const
    {
        KThreeVector a = valueA.GetPosition();
        KThreeVector b = valueB.GetPosition();
        return (a-b).Magnitude();
    }

    double
    KSTrajInterpolatorHermite::DistanceMetric(const KSTrajAdiabaticParticle& valueA, const KSTrajAdiabaticParticle& valueB) const
    {
        KThreeVector a = valueA.GetGuidingCenter();
        KThreeVector b = valueB.GetGuidingCenter();
        return (a-b).Magnitude();
    }

    double
    KSTrajInterpolatorHermite::DistanceMetric(const KSTrajMagneticParticle& valueA, const KSTrajMagneticParticle& valueB) const
    {
        KThreeVector a = valueA.GetPosition();
        KThreeVector b = valueB.GetPosition();
        return (a-b).Magnitude();
    }

    double
    KSTrajInterpolatorHermite::DistanceMetric(const KSTrajElectricParticle& valueA, const KSTrajElectricParticle& valueB) const
    {
        KThreeVector a = valueA.GetPosition();
        KThreeVector b = valueB.GetPosition();
        return (a-b).Magnitude();
    }

    //evaluate cubic hermite basis functions on [0,1]
    void
    KSTrajInterpolatorHermite::CubicHermite(double t, double& h30, double& h31, double& h32, double& h33)
    {
        double t2 = t*t;
        double t3 = t*t2;
        h30 = 1.0 - 3.0*t2 + 2.0*t3;
        h31 = t - 2.0*t2 + t3;
        h32 = t3 - t2;
        h33 = 3.0*t2 - 2.0*t3;
    }

    //evaluate quintic hermite basis functions on [0,1]
    void
    KSTrajInterpolatorHermite::QuinticHermite(double t, double& h50, double& h51, double& h52, double& h53, double& h54, double& h55)
    {
        double t2 = t*t;
        double t3 = t*t2;
        double t4 = t*t3;
        double t5 = t*t4;

        h50 = 1 - 10.0*t3 + 15.0*t4 - 6.0*t5;
        h51 = t - 6.0*t3 + 8.0*t4 - 3.0*t5;
        h52 = 0.5*t2 - 1.5*t3 + 1.5*t4 - 0.5*t5;
        h53 = 0.5*t3 - t4 + 0.5*t5;
        h54 = -4.0*t3 + 7.0*t4 - 3.0*t5;
        h55 = 10.0*t3 - 15.0*t4 + 6.0*t5;
    }


}
