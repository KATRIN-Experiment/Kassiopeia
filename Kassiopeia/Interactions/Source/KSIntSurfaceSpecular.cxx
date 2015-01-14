#include "KSIntSurfaceSpecular.h"

#include "KSInteractionsMessage.h"

#include "KRandom.h"
using katrin::KRandom;

namespace Kassiopeia
{

    KSIntSurfaceSpecular::KSIntSurfaceSpecular() :
            fProbability( .0 ),
            fReflectionLoss( 0. ),
            fTransmissionLoss( 0. )
    {
    }
    KSIntSurfaceSpecular::KSIntSurfaceSpecular( const KSIntSurfaceSpecular& aCopy ) :
            fProbability( aCopy.fProbability ),
            fReflectionLoss( aCopy.fReflectionLoss ),
            fTransmissionLoss( aCopy.fTransmissionLoss )
    {
    }
    KSIntSurfaceSpecular* KSIntSurfaceSpecular::Clone() const
    {
        return new KSIntSurfaceSpecular( *this );
    }
    KSIntSurfaceSpecular::~KSIntSurfaceSpecular()
    {
    }

    void KSIntSurfaceSpecular::ExecuteInteraction( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aQueue )
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
    void KSIntSurfaceSpecular::ExecuteReflection( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& )
    {
        const double tKineticEnergy = anInitialParticle.GetKineticEnergy() - KConst::Q() * fReflectionLoss;
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
            intmsg( eError ) << "surface specular interaction named <" << GetName() << "> was given a particle with neither a surface nor a side set" << eom;
            return;
        }
        KThreeVector tInitialMomentum = anInitialParticle.GetMomentum();
        KThreeVector tInitialNormalMomentum = tInitialMomentum.Dot( tNormal ) * tNormal;
        KThreeVector tInitialTangentMomentum = tInitialMomentum - tInitialNormalMomentum;

        aFinalParticle = anInitialParticle;
        aFinalParticle.SetMomentum( tInitialTangentMomentum - tInitialNormalMomentum );
        aFinalParticle.SetKineticEnergy( tKineticEnergy );

        return;
    }
    void KSIntSurfaceSpecular::ExecuteTransmission( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& )
    {
        const double tKineticEnergy = anInitialParticle.GetKineticEnergy() - KConst::Q() * fTransmissionLoss;

        aFinalParticle = anInitialParticle;
        aFinalParticle.SetKineticEnergy( tKineticEnergy );

        return;
    }

}
