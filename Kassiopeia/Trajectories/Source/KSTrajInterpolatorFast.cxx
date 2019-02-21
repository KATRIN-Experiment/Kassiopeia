#include "KSTrajInterpolatorFast.h"

#include "KSTrajectoriesMessage.h"

namespace Kassiopeia
{

    KSTrajInterpolatorFast::KSTrajInterpolatorFast()
    {
    }
    KSTrajInterpolatorFast::KSTrajInterpolatorFast( const KSTrajInterpolatorFast& ):
        KSComponent()
    {
    }
    KSTrajInterpolatorFast* KSTrajInterpolatorFast::Clone() const
    {
        return new KSTrajInterpolatorFast( *this );
    }
    KSTrajInterpolatorFast::~KSTrajInterpolatorFast()
    {
    }

    void KSTrajInterpolatorFast::Interpolate(double /*aTime*/, const KSTrajExactIntegrator& /*anIntegrator*/, const KSTrajExactDifferentiator& /*aDifferentiator*/, const KSTrajExactParticle& anInitialParticle, const KSTrajExactParticle& aFinalParticle, const double& aTimeStep, KSTrajExactParticle& anIntermediateParticle ) const
    {
        double tInitialTime = anInitialParticle.GetTime();
        double tFinalTime = aFinalParticle.GetTime();
        double tFraction = aTimeStep / (tFinalTime - tInitialTime);
        double tInterpolatedTime = tInitialTime + aTimeStep;

        // interpolate the length
        double tInitialLength = anInitialParticle.GetLength();
        double tFinalLength = aFinalParticle.GetLength();
        double tInterpolatedLength = tInitialLength + tFraction * (tFinalLength - tInitialLength);

        // interpolate the position
        KThreeVector tInitialPosition = anInitialParticle.GetPosition();
        KThreeVector tFinalPosition = aFinalParticle.GetPosition();
        KThreeVector tInterpolatedPosition = tInitialPosition + tFraction * (tFinalPosition - tInitialPosition);

        // interpolate the momentum direction
        KThreeVector tInitialMomentum = anInitialParticle.GetMomentum();
        KThreeVector tFinalMomentum = aFinalParticle.GetMomentum();
        KThreeVector tInitialMomentumDirection = tInitialMomentum.Unit();
        KThreeVector tFinalMomentumDirection = tFinalMomentum.Unit();
        KThreeVector tRotationVector = tInitialMomentumDirection.Cross( tFinalMomentumDirection );
        double tRotationAngle = tFraction * asin( tRotationVector.Magnitude() );
        KThreeVector tRotationAxis = tRotationVector.Unit();
        KThreeVector tInterpolatedMomentumDirection = cos( tRotationAngle ) * tInitialMomentumDirection + sin( tRotationAngle ) * tRotationAxis.Cross( tInitialMomentumDirection );

        // interpolate the momentum magnitude
        double tInitialMomentumMagnitude = tInitialMomentum.Magnitude();
        double tFinalMomentumMagnitude = tFinalMomentum.Magnitude();
        double tInterpolatedMomentumMagnitude = tInitialMomentumMagnitude + tFraction * (tFinalMomentumMagnitude - tInitialMomentumMagnitude);

        anIntermediateParticle = 0.;
        anIntermediateParticle[ 0 ] = tInterpolatedTime;
        anIntermediateParticle[ 1 ] = tInterpolatedLength;
        anIntermediateParticle[ 2 ] = tInterpolatedPosition.X();
        anIntermediateParticle[ 3 ] = tInterpolatedPosition.Y();
        anIntermediateParticle[ 4 ] = tInterpolatedPosition.Z();
        anIntermediateParticle[ 5 ] = tInterpolatedMomentumMagnitude * tInterpolatedMomentumDirection.X();
        anIntermediateParticle[ 6 ] = tInterpolatedMomentumMagnitude * tInterpolatedMomentumDirection.Y();
        anIntermediateParticle[ 7 ] = tInterpolatedMomentumMagnitude * tInterpolatedMomentumDirection.Z();

        return;
    }

    void KSTrajInterpolatorFast::Interpolate(double /*aTime*/, const KSTrajExactSpinIntegrator& /*anIntegrator*/, const KSTrajExactSpinDifferentiator& /*aDifferentiator*/, const KSTrajExactSpinParticle& anInitialParticle, const KSTrajExactSpinParticle& aFinalParticle, const double& aTimeStep, KSTrajExactSpinParticle& anIntermediateParticle ) const
    {
        double tInitialTime = anInitialParticle.GetTime();
        double tFinalTime = aFinalParticle.GetTime();
        double tFraction = aTimeStep / (tFinalTime - tInitialTime);
        double tInterpolatedTime = tInitialTime + aTimeStep;

        // interpolate the length
        double tInitialLength = anInitialParticle.GetLength();
        double tFinalLength = aFinalParticle.GetLength();
        double tInterpolatedLength = tInitialLength + tFraction * (tFinalLength - tInitialLength);

        // interpolate the position
        KThreeVector tInitialPosition = anInitialParticle.GetPosition();
        KThreeVector tFinalPosition = aFinalParticle.GetPosition();
        KThreeVector tInterpolatedPosition = tInitialPosition + tFraction * (tFinalPosition - tInitialPosition);

        // interpolate the momentum direction
        KThreeVector tInitialMomentum = anInitialParticle.GetMomentum();
        KThreeVector tFinalMomentum = aFinalParticle.GetMomentum();
        KThreeVector tInitialMomentumDirection = tInitialMomentum.Unit();
        KThreeVector tFinalMomentumDirection = tFinalMomentum.Unit();
        KThreeVector tRotationVector = tInitialMomentumDirection.Cross( tFinalMomentumDirection );
        double tRotationAngle = tFraction * asin( tRotationVector.Magnitude() );
        KThreeVector tRotationAxis = tRotationVector.Unit();
        KThreeVector tInterpolatedMomentumDirection = cos( tRotationAngle ) * tInitialMomentumDirection + sin( tRotationAngle ) * tRotationAxis.Cross( tInitialMomentumDirection );

        // interpolate the momentum magnitude
        double tInitialMomentumMagnitude = tInitialMomentum.Magnitude();
        double tFinalMomentumMagnitude = tFinalMomentum.Magnitude();
        double tInterpolatedMomentumMagnitude = tInitialMomentumMagnitude + tFraction * (tFinalMomentumMagnitude - tInitialMomentumMagnitude);

        // does not interpolate spin
        trajmsg( eWarning ) << "Fast interpolation does not interpolate spins." << eom;

        anIntermediateParticle = 0.;
        anIntermediateParticle[ 0 ] = tInterpolatedTime;
        anIntermediateParticle[ 1 ] = tInterpolatedLength;
        anIntermediateParticle[ 2 ] = tInterpolatedPosition.X();
        anIntermediateParticle[ 3 ] = tInterpolatedPosition.Y();
        anIntermediateParticle[ 4 ] = tInterpolatedPosition.Z();
        anIntermediateParticle[ 5 ] = tInterpolatedMomentumMagnitude * tInterpolatedMomentumDirection.X();
        anIntermediateParticle[ 6 ] = tInterpolatedMomentumMagnitude * tInterpolatedMomentumDirection.Y();
        anIntermediateParticle[ 7 ] = tInterpolatedMomentumMagnitude * tInterpolatedMomentumDirection.Z();

        return;
    }

    void KSTrajInterpolatorFast::Interpolate(double /*aTime*/, const KSTrajAdiabaticSpinIntegrator& /*anIntegrator*/, const KSTrajAdiabaticSpinDifferentiator& /*aDifferentiator*/, const KSTrajAdiabaticSpinParticle& anInitialParticle, const KSTrajAdiabaticSpinParticle& aFinalParticle, const double& aTimeStep, KSTrajAdiabaticSpinParticle& anIntermediateParticle ) const
    {
        double tInitialTime = anInitialParticle.GetTime();
        double tFinalTime = aFinalParticle.GetTime();
        double tFraction = aTimeStep / (tFinalTime - tInitialTime);
        double tInterpolatedTime = tInitialTime + aTimeStep;

        // interpolate the length
        double tInitialLength = anInitialParticle.GetLength();
        double tFinalLength = aFinalParticle.GetLength();
        double tInterpolatedLength = tInitialLength + tFraction * (tFinalLength - tInitialLength);

        // interpolate the position
        KThreeVector tInitialPosition = anInitialParticle.GetPosition();
        KThreeVector tFinalPosition = aFinalParticle.GetPosition();
        KThreeVector tInterpolatedPosition = tInitialPosition + tFraction * (tFinalPosition - tInitialPosition);

        // interpolate the momentum direction
        KThreeVector tInitialMomentum = anInitialParticle.GetMomentum();
        KThreeVector tFinalMomentum = aFinalParticle.GetMomentum();
        KThreeVector tInitialMomentumDirection = tInitialMomentum.Unit();
        KThreeVector tFinalMomentumDirection = tFinalMomentum.Unit();
        KThreeVector tRotationVector = tInitialMomentumDirection.Cross( tFinalMomentumDirection );
        double tRotationAngle = tFraction * asin( tRotationVector.Magnitude() );
        KThreeVector tRotationAxis = tRotationVector.Unit();
        KThreeVector tInterpolatedMomentumDirection = cos( tRotationAngle ) * tInitialMomentumDirection + sin( tRotationAngle ) * tRotationAxis.Cross( tInitialMomentumDirection );

        // interpolate the momentum magnitude
        double tInitialMomentumMagnitude = tInitialMomentum.Magnitude();
        double tFinalMomentumMagnitude = tFinalMomentum.Magnitude();
        double tInterpolatedMomentumMagnitude = tInitialMomentumMagnitude + tFraction * (tFinalMomentumMagnitude - tInitialMomentumMagnitude);

        // does not interpolate spin
        trajmsg( eWarning ) << "Fast interpolation does not interpolate spins." << eom;

        anIntermediateParticle = 0.;
        anIntermediateParticle[ 0 ] = tInterpolatedTime;
        anIntermediateParticle[ 1 ] = tInterpolatedLength;
        anIntermediateParticle[ 2 ] = tInterpolatedPosition.X();
        anIntermediateParticle[ 3 ] = tInterpolatedPosition.Y();
        anIntermediateParticle[ 4 ] = tInterpolatedPosition.Z();
        anIntermediateParticle[ 5 ] = tInterpolatedMomentumMagnitude * tInterpolatedMomentumDirection.X();
        anIntermediateParticle[ 6 ] = tInterpolatedMomentumMagnitude * tInterpolatedMomentumDirection.Y();
        anIntermediateParticle[ 7 ] = tInterpolatedMomentumMagnitude * tInterpolatedMomentumDirection.Z();

        return;
    }

    void KSTrajInterpolatorFast::Interpolate(double /*aTime*/, const KSTrajAdiabaticIntegrator& /*anIntegrator*/, const KSTrajAdiabaticDifferentiator& /*aDifferentiator*/, const KSTrajAdiabaticParticle& anInitialParticle, const KSTrajAdiabaticParticle& aFinalParticle, const double& aTimeStep, KSTrajAdiabaticParticle& anIntermediateParticle ) const
    {
        double tInitialTime = anInitialParticle.GetTime();
        double tFinalTime = aFinalParticle.GetTime();
        double tFraction = aTimeStep / (tFinalTime - tInitialTime);
        double tInterpolatedTime = tInitialTime + aTimeStep;

        // interpolate the length
        double tInitialLength = anInitialParticle.GetLength();
        double tFinalLength = aFinalParticle.GetLength();
        double tInterpolatedLength = tInitialLength + tFraction * (tFinalLength - tInitialLength);

        // interpolate the guiding center
        KThreeVector tInitialGuidingCenter = anInitialParticle.GetGuidingCenter();
        KThreeVector tFinalGuidingCenter = aFinalParticle.GetGuidingCenter();
        KThreeVector tInterpolatedGuidingCenter = tInitialGuidingCenter + tFraction * (tFinalGuidingCenter - tInitialGuidingCenter);

        //interpolate the long and trans momentum
        double tInitialLongMomentum = anInitialParticle.GetLongMomentum();
        double tFinalLongMomentum = aFinalParticle.GetLongMomentum();
        double tInterpolatedLongMomentum = tInitialLongMomentum + tFraction * (tFinalLongMomentum - tInitialLongMomentum);

        double tInitialTransMomentum = anInitialParticle.GetTransMomentum();
        double tFinalTransMomentum = aFinalParticle.GetTransMomentum();
        double tInterpolatedTransMomentum = tInitialTransMomentum + tFraction * (tFinalTransMomentum - tInitialTransMomentum);

        //interpolate phase
        //if no gyration term is used, this phase interpolation is wrong (should stay at 0 instead). But there is no way I can check this from here right now
        double tInitialPhase = anInitialParticle.GetPhase();
        double tFinalPhase = aFinalParticle.GetPhase();
        double tInitialCyclotronfrequency = anInitialParticle.GetCyclotronFrequency();
        double tFinalCyclotronfrequency = aFinalParticle.GetCyclotronFrequency();
        double tInterpolatedCylcontronfrequency = tInitialCyclotronfrequency + tFraction * (tFinalCyclotronfrequency - tInitialCyclotronfrequency);
        double tNTurns = tInterpolatedCylcontronfrequency * ( tFinalTime - tInitialTime );
        int tNTurnsInt = floor( tNTurns );
        tFinalPhase += tNTurnsInt * 2.*KConst::Pi();
        double tInterpolatedPhase = tInitialPhase + tFraction * (tFinalPhase - tInitialPhase);

        //interpolate alpha and beta
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

        anIntermediateParticle = 0.;
        anIntermediateParticle[ 0 ] = tInterpolatedTime;
        anIntermediateParticle[ 1 ] = tInterpolatedLength;
        anIntermediateParticle[ 2 ] = tInterpolatedGuidingCenter.X();
        anIntermediateParticle[ 3 ] = tInterpolatedGuidingCenter.Y();
        anIntermediateParticle[ 4 ] = tInterpolatedGuidingCenter.Z();
        anIntermediateParticle[ 5 ] = tInterpolatedLongMomentum;
        anIntermediateParticle[ 6 ] = tInterpolatedTransMomentum;
        anIntermediateParticle[ 7 ] = tInterpolatedPhase;

        anIntermediateParticle.SetAlpha( tInterpolatedAlpha );
        anIntermediateParticle.SetBeta( tInterpolatedBeta );

    }

    void KSTrajInterpolatorFast::Interpolate(double /*aTime*/, const KSTrajMagneticIntegrator& /*anIntegrator*/, const KSTrajMagneticDifferentiator& /*aDifferentiator*/, const KSTrajMagneticParticle& anInitialParticle, const KSTrajMagneticParticle& aFinalParticle, const double& aTimeStep, KSTrajMagneticParticle& anIntermediateParticle ) const
    {
        double tInitialTime = anInitialParticle.GetTime();
        double tFinalTime = aFinalParticle.GetTime();
        double tFraction = aTimeStep / (tFinalTime - tInitialTime);
        double tInterpolatedTime = tInitialTime + aTimeStep;

        // interpolate the length
        double tInitialLength = anInitialParticle.GetLength();
        double tFinalLength = aFinalParticle.GetLength();
        double tInterpolatedLength = tInitialLength + tFraction * (tFinalLength - tInitialLength);

        // interpolate the position
        KThreeVector tInitialPosition = anInitialParticle.GetPosition();
        KThreeVector tFinalPosition = aFinalParticle.GetPosition();
        KThreeVector tInterpolatedPosition = tInitialPosition + tFraction * (tFinalPosition - tInitialPosition);

        anIntermediateParticle = 0.;
        anIntermediateParticle[ 0 ] = tInterpolatedTime;
        anIntermediateParticle[ 1 ] = tInterpolatedLength;
        anIntermediateParticle[ 2 ] = tInterpolatedPosition.X();
        anIntermediateParticle[ 3 ] = tInterpolatedPosition.Y();
        anIntermediateParticle[ 4 ] = tInterpolatedPosition.Z();
    }

    void KSTrajInterpolatorFast::Interpolate(double /*aTime*/, const KSTrajElectricIntegrator& /*anIntegrator*/, const KSTrajElectricDifferentiator& /*aDifferentiator*/, const KSTrajElectricParticle& anInitialParticle, const KSTrajElectricParticle& aFinalParticle, const double& aTimeStep, KSTrajElectricParticle& anIntermediateParticle ) const
    {
        double tInitialTime = anInitialParticle.GetTime();
        double tFinalTime = aFinalParticle.GetTime();
        double tFraction = aTimeStep / (tFinalTime - tInitialTime);
        double tInterpolatedTime = tInitialTime + aTimeStep;

        // interpolate the length
        double tInitialLength = anInitialParticle.GetLength();
        double tFinalLength = aFinalParticle.GetLength();
        double tInterpolatedLength = tInitialLength + tFraction * (tFinalLength - tInitialLength);

        // interpolate the position
        KThreeVector tInitialPosition = anInitialParticle.GetPosition();
        KThreeVector tFinalPosition = aFinalParticle.GetPosition();
        KThreeVector tInterpolatedPosition = tInitialPosition + tFraction * (tFinalPosition - tInitialPosition);

        anIntermediateParticle = 0.;
        anIntermediateParticle[ 0 ] = tInterpolatedTime;
        anIntermediateParticle[ 1 ] = tInterpolatedLength;
        anIntermediateParticle[ 2 ] = tInterpolatedPosition.X();
        anIntermediateParticle[ 3 ] = tInterpolatedPosition.Y();
        anIntermediateParticle[ 4 ] = tInterpolatedPosition.Z();
    }

    void KSTrajInterpolatorFast::GetPiecewiseLinearApproximation
    (
        double /*aTolerance*/,
        unsigned int /*nMaxSegments*/,
        double /*anInitialTime*/,
        double /*aFinalTime*/,
        const KSTrajExactIntegrator& /*anIntegrator*/,
        const KSTrajExactDifferentiator& /*aDifferentiator*/,
        const KSTrajExactParticle& anInitialParticle,
        const KSTrajExactParticle& aFinalParticle,
        std::vector<KSTrajExactParticle>* interpolatedValues
    )
    const
    {
        interpolatedValues->clear();
        interpolatedValues->push_back(anInitialParticle);
        interpolatedValues->push_back(aFinalParticle);
    }

    void KSTrajInterpolatorFast::GetPiecewiseLinearApproximation
    (
        double /*aTolerance*/,
        unsigned int /*nMaxSegments*/,
        double /*anInitialTime*/,
        double /*aFinalTime*/,
        const KSTrajExactSpinIntegrator& /*anIntegrator*/,
        const KSTrajExactSpinDifferentiator& /*aDifferentiator*/,
        const KSTrajExactSpinParticle& anInitialParticle,
        const KSTrajExactSpinParticle& aFinalParticle,
        std::vector<KSTrajExactSpinParticle>* interpolatedValues
    )
    const
    {
        interpolatedValues->clear();
        interpolatedValues->push_back(anInitialParticle);
        interpolatedValues->push_back(aFinalParticle);
    }

    void KSTrajInterpolatorFast::GetPiecewiseLinearApproximation
    (
        double /*aTolerance*/,
        unsigned int /*nMaxSegments*/,
        double /*anInitialTime*/,
        double /*aFinalTime*/,
        const KSTrajAdiabaticSpinIntegrator& /*anIntegrator*/,
        const KSTrajAdiabaticSpinDifferentiator& /*aDifferentiator*/,
        const KSTrajAdiabaticSpinParticle& anInitialParticle,
        const KSTrajAdiabaticSpinParticle& aFinalParticle,
        std::vector<KSTrajAdiabaticSpinParticle>* interpolatedValues
    )
    const
    {
        interpolatedValues->clear();
        interpolatedValues->push_back(anInitialParticle);
        interpolatedValues->push_back(aFinalParticle);
    }

    void KSTrajInterpolatorFast::GetPiecewiseLinearApproximation
    (
        double /*aTolerance*/,
        unsigned int /*nMaxSegments*/,
        double /*anInitialTime*/,
        double /*aFinalTime*/,
        const KSTrajAdiabaticIntegrator& /*anIntegrator*/,
        const KSTrajAdiabaticDifferentiator& /*aDifferentiator*/,
        const KSTrajAdiabaticParticle& anInitialParticle,
        const KSTrajAdiabaticParticle& aFinalParticle,
        std::vector<KSTrajAdiabaticParticle>* interpolatedValues
    )
    const
    {
        interpolatedValues->clear();
        interpolatedValues->push_back(anInitialParticle);
        interpolatedValues->push_back(aFinalParticle);
    }

    void KSTrajInterpolatorFast::GetPiecewiseLinearApproximation
    (
        double /*aTolerance*/,
        unsigned int /*nMaxSegments*/,
        double /*anInitialTime*/,
        double /*aFinalTime*/,
        const KSTrajMagneticIntegrator& /*anIntegrator*/,
        const KSTrajMagneticDifferentiator& /*aDifferentiator*/,
        const KSTrajMagneticParticle& anInitialParticle,
        const KSTrajMagneticParticle& aFinalParticle,
        std::vector<KSTrajMagneticParticle>* interpolatedValues
    )
    const
    {
        interpolatedValues->clear();
        interpolatedValues->push_back(anInitialParticle);
        interpolatedValues->push_back(aFinalParticle);
    }


    void KSTrajInterpolatorFast::GetPiecewiseLinearApproximation
    (
        double /*aTolerance*/,
        unsigned int /*nMaxSegments*/,
        double /*anInitialTime*/,
        double /*aFinalTime*/,
        const KSTrajElectricIntegrator& /*anIntegrator*/,
        const KSTrajElectricDifferentiator& /*aDifferentiator*/,
        const KSTrajElectricParticle& anInitialParticle,
        const KSTrajElectricParticle& aFinalParticle,
        std::vector<KSTrajElectricParticle>* interpolatedValues
    )
    const
    {
        interpolatedValues->clear();
        interpolatedValues->push_back(anInitialParticle);
        interpolatedValues->push_back(aFinalParticle);
    }


}
