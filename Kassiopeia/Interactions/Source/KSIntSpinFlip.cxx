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
        //std::cout << "BUILT\n";
    }
    KSIntSpinFlip::KSIntSpinFlip( const KSIntSpinFlip& aCopy ) :
            KSComponent(),
            KSComponentTemplate< KSIntSpinFlip, KSSpaceInteraction >( aCopy )
    {
        //std::cout << "BUILT\n";
    }
    KSIntSpinFlip* KSIntSpinFlip::Clone() const
    {
        //std::cout << "BUILT\n";
        return new KSIntSpinFlip( *this );
    }
    KSIntSpinFlip::~KSIntSpinFlip()
    {
//        for( unsigned int tIndex = 0; tIndex < fCalculators.size(); tIndex++ )
//        {
//            delete (fCalculators.at( tIndex ));
//        }
//        fCalculators.clear();
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

      //std::cout << "GradBMagnitude: " << GradBMagnitude << "\tGradBDirection: " << GradBDirection << "\nBDiretionDot: " << BDirectionDot << "\tCycleFlipProbability: " << CycleFlipProbability << "\n";
      //std::cout << "TransitionRate: " << aTransitionRate << "\n";
      //std::cout << "CALC RATE\n";
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
        //std::cout << "Flip Time: " << FlipTime << "\tTrajectoryTimeStep: " << aTrajectoryTimeStep << "\n";
        if ( std::isnan( FlipTime ) )
        {
            FlipTime = numeric_limits< double >::max();
            //std::cout << "nan\n";
        }

        if( FlipTime > aTrajectoryTimeStep )
        {
            anInteractionParticle = aTrajectoryFinalParticle;
            aTimeStep = aTrajectoryTimeStep;
            aFlag = false;
            //std::cout << "shouldn't flip\tTimeStep: " << aTimeStep << "\n";
        }
        else
        {
            anInteractionParticle = aTrajectoryInitialParticle;
            aTrajectory.ExecuteTrajectory( FlipTime, anInteractionParticle );
            aTimeStep = FlipTime;
            aFlag = true;
            //std::cout << "should flip\tTimeStep: " << aTimeStep << "\n";
        }

        //std::cout << "CALC INT\n";
    }

    void KSIntSpinFlip::ExecuteInteraction( const KSParticle& anInteractionParticle, KSParticle& aFinalParticle, KSParticleQueue& /*aSecondaries*/ ) const
    {
        aFinalParticle = anInteractionParticle;
        aFinalParticle.SetAlignedSpin( -1 * aFinalParticle.GetAlignedSpin() );
        aFinalParticle.SetSpinAngle( aFinalParticle.GetSpinAngle() + KConst::Pi() );

        //std::cout << "Time: " << aFinalParticle.GetTime() << "\tPosition: " << aFinalParticle.GetPosition() << "\n";
        //std::cout << "FLIP\n";
    }

}
