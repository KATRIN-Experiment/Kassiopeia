#include "KSIntSurfaceUCN.h"

#include "KSInteractionsMessage.h"

#include "KRandom.h"
using katrin::KRandom;

namespace Kassiopeia
{

    KSIntSurfaceUCN::KSIntSurfaceUCN() :
            fProbability( .0 ),
            fSpinFlipProbability( 0. )
    {
    }
    KSIntSurfaceUCN::KSIntSurfaceUCN( const KSIntSurfaceUCN& aCopy ) :
            KSComponent(),
            fProbability( aCopy.fProbability ),
            fSpinFlipProbability( aCopy.fSpinFlipProbability )
    {
    }
    KSIntSurfaceUCN* KSIntSurfaceUCN::Clone() const
    {
        return new KSIntSurfaceUCN( *this );
    }
    KSIntSurfaceUCN::~KSIntSurfaceUCN()
    {
    }

    void KSIntSurfaceUCN::ExecuteInteraction( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aQueue )
    {
        double tChoice = KRandom::GetInstance().Uniform( 0., 1. );
        if( tChoice < fProbability )
        {
            ExecuteTransmission( anInitialParticle, aFinalParticle, aQueue );
        }
        else
        {
            ExecuteReflection( anInitialParticle, aFinalParticle, aQueue );
        }
        return;
    }
    void KSIntSurfaceUCN::ExecuteReflection( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& )
    {
        KThreeVector tNormal;
        if( anInitialParticle.GetCurrentSurface() != NULL )
        {
            tNormal = anInitialParticle.GetCurrentSurface()->Normal( anInitialParticle.GetPosition() );
        }
        else if( anInitialParticle.GetCurrentSide() != NULL )
        {
            tNormal = anInitialParticle.GetCurrentSide()->Normal( anInitialParticle.GetPosition() );
        }
        else
        {
            intmsg( eError ) << "surface UCN interaction named <" << GetName() << "> was given a particle with neither a surface nor a side set" << eom;
            return;
        }
        KThreeVector tInitialMomentum = anInitialParticle.GetMomentum();
        KThreeVector tInitialNormalMomentum = tInitialMomentum.Dot( tNormal ) * tNormal;
        KThreeVector tInitialTangentMomentum = tInitialMomentum - tInitialNormalMomentum;

        KThreeVector tSpin = aFinalParticle.GetSpin();
        double tAlignedSpin = aFinalParticle.GetAlignedSpin();
        double tSpinAngle = aFinalParticle.GetSpinAngle();

        double tChoice = KRandom::GetInstance().Uniform( 0., 1. );
        if( tChoice < 2*fSpinFlipProbability ) // there's a 50-50 chance of getting the old spin after measurement, hence the 2*
        {
            bool done = false;
            while ( !done ){
                double tx = KRandom::GetInstance().Uniform( -1., 1. );
                double ty = KRandom::GetInstance().Uniform( -1., 1. );
                double tz = KRandom::GetInstance().Uniform( -1., 1. );
                if ( tx*tx + ty*ty + tz*tz < 1. ){
                      tSpin = KThreeVector( tx, ty, tz );
                      tSpin = tSpin/tSpin.Magnitude();
                      tAlignedSpin = tSpin.Dot( anInitialParticle.GetMagneticField() ) / anInitialParticle.GetMagneticField().Magnitude();
                      tSpinAngle = KRandom::GetInstance().Uniform( 0., 180. );
                      done = true;
                }
            }
        }

        aFinalParticle = anInitialParticle;
        aFinalParticle.SetMomentum( tInitialTangentMomentum - tInitialNormalMomentum );

        // spin changes need to happen aftet SetMomentum to make Spin0 correct

        aFinalParticle.SetInitialSpin( tSpin );
        aFinalParticle.SetAlignedSpin( tAlignedSpin );
        aFinalParticle.SetSpinAngle( tSpinAngle );

        return;
    }
    void KSIntSurfaceUCN::ExecuteTransmission( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& )
    {

        KThreeVector tSpin = aFinalParticle.GetSpin();
        double tAlignedSpin = aFinalParticle.GetAlignedSpin();
        double tSpinAngle = aFinalParticle.GetSpinAngle();

        aFinalParticle = anInitialParticle;

        aFinalParticle.SetInitialSpin( tSpin );
        aFinalParticle.SetAlignedSpin( tAlignedSpin );
        aFinalParticle.SetSpinAngle( tSpinAngle );

        return;
    }

}
