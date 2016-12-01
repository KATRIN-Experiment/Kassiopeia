#include "KSIntSurfaceDiffuse.h"

#include "KSInteractionsMessage.h"

#include "KRandom.h"
using katrin::KRandom;

#include "KConst.h"
using katrin::KConst;

#include <iostream>
#include <cmath>

namespace Kassiopeia
{

    KSIntSurfaceDiffuse::KSIntSurfaceDiffuse() :
            fProbability( .0 ),
            fReflectionLoss( 0. ),
            fTransmissionLoss( 0. ),
            fReflectionLossFraction( 0. ),
            fTransmissionLossFraction( 0. ),
            fUseRelativeLoss(false)
    {
    }
    KSIntSurfaceDiffuse::KSIntSurfaceDiffuse( const KSIntSurfaceDiffuse& aCopy ) :
            KSComponent(),
            fProbability( aCopy.fProbability ),
            fReflectionLoss( aCopy.fReflectionLoss ),
            fTransmissionLoss( aCopy.fTransmissionLoss ),
            fReflectionLossFraction( aCopy.fReflectionLossFraction ),
            fTransmissionLossFraction( aCopy.fTransmissionLossFraction ),
            fUseRelativeLoss( aCopy.fUseRelativeLoss)
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
        double tKineticEnergy = anInitialParticle.GetKineticEnergy();

        if(fUseRelativeLoss)
        {
            tKineticEnergy *= (1.0 - fReflectionLossFraction);
        }
        else
        {
            tKineticEnergy -= std::fabs(KConst::Q() * fReflectionLoss);
        }

        //prevent kinetic energy from going negative
        if(tKineticEnergy < 0.0)
        {
            intmsg( eError ) << "surface diffuse interaction named <" << GetName() << "> tried to give a particle a negative kinetic energy." << eom;
            return;
        }

        //generate angles for diffuse 'Lambertian' reflection direction
        double tAzimuthalAngle = KRandom::GetInstance().Uniform( 0., 2. * KConst::Pi() );
        double tSinTheta = KRandom::GetInstance().Uniform( 0., 1. );
        double tCosTheta = std::sqrt( (1.0 - tSinTheta)*(1.0 + tSinTheta) );

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
        double tMomentumMagnitude = tInitialMomentum.Magnitude();

        KThreeVector tInitialNormalMomentum = tInitialMomentum.Dot( tNormal ) * tNormal;
        KThreeVector tInitialTangentMomentum = tInitialMomentum - tInitialNormalMomentum;
        KThreeVector tInitialOrthogonalMomentum = tInitialTangentMomentum.Cross( tInitialNormalMomentum.Unit() );

        //define reflected direction
        KThreeVector tReflectedDirection = -1.0*tCosTheta*tInitialNormalMomentum.Unit();
        tReflectedDirection += tSinTheta*std::cos(tAzimuthalAngle)*tInitialTangentMomentum.Unit();
        tReflectedDirection += tSinTheta*std::sin(tAzimuthalAngle)*tInitialOrthogonalMomentum.Unit();

        KThreeVector tReflectedMomentum = tMomentumMagnitude*tReflectedDirection;

        aFinalParticle = anInitialParticle;
        aFinalParticle.SetMomentum( tReflectedMomentum );
        aFinalParticle.SetKineticEnergy( tKineticEnergy );

        return;
    }
    void KSIntSurfaceDiffuse::ExecuteTransmission( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& )
    {
        double tKineticEnergy = anInitialParticle.GetKineticEnergy();

        if(fUseRelativeLoss)
        {
            tKineticEnergy *= (1.0 - fTransmissionLossFraction);
        }
        else
        {
            tKineticEnergy -= std::fabs(KConst::Q() * fTransmissionLoss);
        }

        //prevent kinetic energy from going negative
        if(tKineticEnergy < 0.0)
        {
            intmsg( eError ) << "surface diffuse interaction named <" << GetName() << "> tried to give a particle a negative kinetic energy." << eom;
            return;
        }

        aFinalParticle = anInitialParticle;
        aFinalParticle.SetKineticEnergy( tKineticEnergy );

        return;
    }

}
