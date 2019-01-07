  #include "KSIntSpinFlip.h"
#include "KSIntDensity.h"
#include "KSInteractionsMessage.h"

#include "KRandom.h"
using katrin::KRandom;

#include <limits>
using std::numeric_limits;

#include <math.h>

namespace Kassiopeia
{

    KSIntSpinFlip::KSIntSpinFlip()
    {
    }
    KSIntSpinFlip::KSIntSpinFlip( const KSIntSpinFlip& aCopy ) :
            KSComponent(),
            KSComponentTemplate< KSIntSpinFlip, KSSpaceInteraction >( aCopy )
    {
    }
    KSIntSpinFlip* KSIntSpinFlip::Clone() const
    {
        return new KSIntSpinFlip( *this );
    }
    KSIntSpinFlip::~KSIntSpinFlip()
    {
    }

    void KSIntSpinFlip::CalculateTransitionRate( const KSParticle& aParticle, double& aTransitionRate )
    {
      KThreeVector GradBMagnitude = aParticle.GetMagneticGradient() * aParticle.GetMagneticField() / aParticle.GetMagneticField().Magnitude();
      KThreeMatrix GradBDirection = aParticle.GetMagneticGradient() / aParticle.GetMagneticField().Magnitude() - KThreeMatrix::OuterProduct( aParticle.GetMagneticField(), GradBMagnitude ) / aParticle.GetMagneticField().Magnitude() / aParticle.GetMagneticField().Magnitude();
      KThreeVector BDirectionDot = aParticle.GetVelocity() * GradBDirection;

      double CycleFlipProbability = 1 / aParticle.GetGyromagneticRatio() / aParticle.GetGyromagneticRatio() / aParticle.GetMagneticField().MagnitudeSquared()
                    * BDirectionDot.MagnitudeSquared()
                    * sin( KConst::Pi() * ( 1 - aParticle.GetAlignedSpin() ) ) * sin( KConst::Pi() * ( 1 - aParticle.GetAlignedSpin() ) );

      aTransitionRate = std::fabs( CycleFlipProbability * aParticle.GetGyromagneticRatio() * aParticle.GetMagneticField().Magnitude() / 2 / KConst::Pi() );
    }

    void KSIntSpinFlip::CalculateInteraction(
            const KSTrajectory& aTrajectory,
            const KSParticle& aTrajectoryInitialParticle,
            const KSParticle& aTrajectoryFinalParticle,
            const KThreeVector& /*aTrajectoryCenter*/,
            const double& /*aTrajectoryRadius*/,
            const double& aTrajectoryTimeStep,
            KSParticle& anInteractionParticle,
            double& aTimeStep,
            bool& aFlag
            )
    {
        double TransitionRate = 0.0;
        CalculateTransitionRate( aTrajectoryInitialParticle, TransitionRate );
        double tProbability = KRandom::GetInstance().Uniform( 0., 1. );
        double FlipTime = -1. * log( 1. - tProbability ) / TransitionRate;
        if ( std::isnan( FlipTime ) )
        {
            FlipTime = numeric_limits< double >::max();
        }

        if( FlipTime > aTrajectoryTimeStep )
        {
            anInteractionParticle = aTrajectoryFinalParticle;
            aTimeStep = aTrajectoryTimeStep;
            aFlag = false;
        }
        else
        {
            anInteractionParticle = aTrajectoryInitialParticle;
            aTrajectory.ExecuteTrajectory( FlipTime, anInteractionParticle );
            aTimeStep = FlipTime;
            aFlag = true;
        }
    }

    void KSIntSpinFlip::ExecuteInteraction( const KSParticle& anInteractionParticle, KSParticle& aFinalParticle, KSParticleQueue& /*aSecondaries*/ ) const
    {
        aFinalParticle = anInteractionParticle;
        aFinalParticle.SetAlignedSpin( -1 * aFinalParticle.GetAlignedSpin() );
        aFinalParticle.SetSpinAngle( aFinalParticle.GetSpinAngle() + KConst::Pi() );
    }

}
