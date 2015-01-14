#include "KSIntSurfaceDiffuse.h"

#include "KSInteractionsMessage.h"

#include "KRandom.h"
using katrin::KRandom;

#include "KConst.h"
using katrin::KConst;

namespace Kassiopeia
{

    KSIntSurfaceDiffuse::KSIntSurfaceDiffuse() :
            fProbability( .0 ),
            fReflectionLoss( 0. ),
            fTransmissionLoss( 0. )
    {
    }
    KSIntSurfaceDiffuse::KSIntSurfaceDiffuse( const KSIntSurfaceDiffuse& aCopy ) :
            fProbability( aCopy.fProbability ),
            fReflectionLoss( aCopy.fReflectionLoss ),
            fTransmissionLoss( aCopy.fTransmissionLoss )
    {
    }
    KSIntSurfaceDiffuse* KSIntSurfaceDiffuse::Clone() const
    {
        return new KSIntSurfaceDiffuse( *this );
    }
    KSIntSurfaceDiffuse::~KSIntSurfaceDiffuse()
    {
    }

    void KSIntSurfaceDiffuse::ExecuteInteraction( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aQueue )
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
    void KSIntSurfaceDiffuse::ExecuteReflection( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& )
    {
        const double tKineticEnergy = anInitialParticle.GetKineticEnergy() - KConst::Q() * fReflectionLoss;
        const double tAngle = KRandom::GetInstance().Uniform( 0., 2. * KConst::Pi() );
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
            intmsg( eError ) << "surface diffuse interaction named <" << GetName() << "> was given a particle with neither a surface nor a side set" << eom;
            return;
        }
        KThreeVector tInitialMomentum = anInitialParticle.GetMomentum();
        KThreeVector tInitialNormalMomentum = tInitialMomentum.Dot( tNormal ) * tNormal;
        KThreeVector tInitialTangentMomentum = tInitialMomentum - tInitialNormalMomentum;
        KThreeVector tInitialOrthogonalMomentum = tInitialTangentMomentum.Cross( tInitialNormalMomentum.Unit() );
        aFinalParticle = anInitialParticle;
        aFinalParticle.SetMomentum( cos( tAngle ) * tInitialTangentMomentum + sin( tAngle ) * tInitialOrthogonalMomentum - tInitialNormalMomentum );
        aFinalParticle.SetKineticEnergy( tKineticEnergy );

        return;
    }
    void KSIntSurfaceDiffuse::ExecuteTransmission( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& )
    {
        const double tKineticEnergy = anInitialParticle.GetKineticEnergy() - KConst::Q() * fTransmissionLoss;

        aFinalParticle = anInitialParticle;
        aFinalParticle.SetKineticEnergy( tKineticEnergy );

        return;
    }

}


