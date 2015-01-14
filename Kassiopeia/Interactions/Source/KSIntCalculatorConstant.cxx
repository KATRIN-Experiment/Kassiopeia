#include "KSIntCalculatorConstant.h"

#include "KRandom.h"
using katrin::KRandom;

#include "KConst.h"
using katrin::KConst;

namespace Kassiopeia
{

    KSIntCalculatorConstant::KSIntCalculatorConstant() :
            fCrossSection( 0. )
    {
    }
    KSIntCalculatorConstant::KSIntCalculatorConstant( const KSIntCalculatorConstant& aCopy ) :
            fCrossSection( aCopy.fCrossSection )
    {
    }
    KSIntCalculatorConstant* KSIntCalculatorConstant::Clone() const
    {
        return new KSIntCalculatorConstant( *this );
    }
    KSIntCalculatorConstant::~KSIntCalculatorConstant()
    {
    }

    void KSIntCalculatorConstant::CalculateCrossSection( const KSParticle& /*aParticle*/, double& aCrossSection )
    {
        aCrossSection = fCrossSection;
        return;
    }
    void KSIntCalculatorConstant::ExecuteInteraction( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& /*aSecondaries*/)
    {
        double tTime = anInitialParticle.GetTime();
        KThreeVector tPosition = anInitialParticle.GetPosition();
        KThreeVector tMomentum = anInitialParticle.GetMomentum();

        double tTheta = acos( KRandom::GetInstance().Uniform( -1., 1. ) );
        double tPhi = KRandom::GetInstance().Uniform( 0., 2. * KConst::Pi() );
        tMomentum.SetPolarAngle( tTheta );
        tMomentum.SetAzimuthalAngle( tPhi );

        aFinalParticle.SetTime( tTime );
        aFinalParticle.SetPosition( tPosition );
        aFinalParticle.SetMomentum( tMomentum );
        aFinalParticle.SetLabel( GetName() );

        return;
    }

}

