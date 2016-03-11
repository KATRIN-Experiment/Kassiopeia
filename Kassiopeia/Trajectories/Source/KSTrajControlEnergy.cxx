#include "KSTrajControlEnergy.h"
#include "KSTrajectoriesMessage.h"

#include <limits>

namespace Kassiopeia
{

    KSTrajControlEnergy::KSTrajControlEnergy() :
            fLowerLimit( 1.e-12 ),  // energy
            fUpperLimit( 1.e-10 ),  // energy
            fMinLength( std::numeric_limits< double >::min() ),  // length
            fMaxLength( std::numeric_limits< double >::max() ),  // length
            fInitialStep( 1e-10 ),  // time
            fAdjustmentFactorUp( 0.5 ),  // factor
            fAdjustmentFactorDown( 0.5 ),  // factor
            fTimeStep( 0. ),  // time
            fFirstStep( true )  // flag
    {
    }
    KSTrajControlEnergy::KSTrajControlEnergy( const KSTrajControlEnergy& aCopy ) :
            KSComponent(),
            fLowerLimit( aCopy.fLowerLimit ),
            fUpperLimit( aCopy.fUpperLimit ),
            fMinLength( aCopy.fMinLength ),
            fMaxLength( aCopy.fMaxLength ),
            fInitialStep( aCopy.fInitialStep ),
            fAdjustmentFactorUp( aCopy.fAdjustmentFactorUp ),
            fAdjustmentFactorDown( aCopy.fAdjustmentFactorDown ),
            fTimeStep( aCopy.fTimeStep ),
            fFirstStep( aCopy.fFirstStep )
    {
    }
    KSTrajControlEnergy* KSTrajControlEnergy::Clone() const
    {
        return new KSTrajControlEnergy( *this );
    }
    KSTrajControlEnergy::~KSTrajControlEnergy()
    {
    }
    void KSTrajControlEnergy::ActivateObject()
    {
        trajmsg_debug( "stepsize energy resetting, adjustment factor is <" << fAdjustmentFactorUp << "/" << fAdjustmentFactorDown << ">" << eom );
        fFirstStep = true;
        fTimeStep = 0.;
        return;
    }

    void KSTrajControlEnergy::Calculate( const KSTrajExactParticle& /*aParticle*/, double& aValue )
    {
        if ( (fFirstStep == true) || (fTimeStep == 0.) )
        {
            trajmsg_debug( "stepsize energy on first step" << eom );
            fTimeStep = fInitialStep;
            fFirstStep = false;
        }

        trajmsg_debug( "stepsize energy suggesting <" << fTimeStep << ">" << eom );
        aValue = fTimeStep;
        return;
    }
    void KSTrajControlEnergy::Check( const KSTrajExactParticle& anInitialParticle, const KSTrajExactParticle& aFinalParticle, const KSTrajExactError&, bool& aFlag )
    {
        fTimeStep = aFinalParticle.GetTime() - anInitialParticle.GetTime();

        double tFinalEnergy = (aFinalParticle.GetKineticEnergy() + aFinalParticle.GetCharge() * aFinalParticle.GetElectricPotential());
        double tInitialEnergy = (anInitialParticle.GetKineticEnergy() + anInitialParticle.GetCharge() * anInitialParticle.GetElectricPotential());
        double tNormalization = tFinalEnergy + tInitialEnergy;
        double tEnergyViolation = 2. * fabs( (tFinalEnergy / tNormalization) - (tInitialEnergy / tNormalization) )  ;

        if( tEnergyViolation< fLowerLimit )
        {
            trajmsg_debug( "stepsize energy increasing stepsize at violation <" << tEnergyViolation << ">" << eom) ;
            fTimeStep = (1. + fAdjustmentFactorUp) * fTimeStep;
            aFlag = false;
            return;
        }
        if( tEnergyViolation > fUpperLimit )
        {
            trajmsg_debug( "stepsize energy decreasing stepsize at violation <" << tEnergyViolation << ">" << eom );
            fTimeStep = (1. - fAdjustmentFactorDown) * fTimeStep;
            aFlag = false;
            return;
        }

        double tStepLength = aFinalParticle.GetLength() - anInitialParticle.GetLength();
        // particle's speed is calculated only when needed since GetVelocity() is a rather expensive operation

        if( tStepLength < fMinLength )
        {
            trajmsg_debug( "stepsize energy increasing stepsize at step length <" << tStepLength << ">" << eom) ;
            fTimeStep = (1. + fAdjustmentFactorUp) * fMinLength / anInitialParticle.GetVelocity().Magnitude();
            aFlag = false;
            return;
        }
        if( tStepLength > fMaxLength )
        {
            trajmsg_debug( "stepsize energy decreasing stepsize at step length <" << tStepLength << ">" << eom) ;
            fTimeStep = (1. - fAdjustmentFactorDown) * fMaxLength / anInitialParticle.GetVelocity().Magnitude();
            aFlag = false;
            return;
        }

        trajmsg_debug( "stepsize energy keeping stepsize at violation <" << tEnergyViolation << ">" << eom) ;
        return;
    }

    void KSTrajControlEnergy::Calculate( const KSTrajAdiabaticParticle& /*aParticle*/, double& aValue )
    {
        if ( (fFirstStep == true) || (fTimeStep == 0.) )
        {
            trajmsg_debug( "stepsize energy on first step" << eom );
            fTimeStep = fInitialStep;;
            fFirstStep = false;
        }

        trajmsg_debug( "stepsize energy suggesting <" << fTimeStep << ">" << eom );
        aValue = fTimeStep;
        return;
    }
    void KSTrajControlEnergy::Check( const KSTrajAdiabaticParticle& anInitialParticle, const KSTrajAdiabaticParticle& aFinalParticle, const KSTrajAdiabaticError&, bool& aFlag )
    {
        fTimeStep = aFinalParticle.GetTime() - anInitialParticle.GetTime();

        double tFinalEnergy = (aFinalParticle.GetKineticEnergy() + aFinalParticle.GetCharge() * aFinalParticle.GetElectricPotential());
        double tInitialEnergy = (anInitialParticle.GetKineticEnergy() + anInitialParticle.GetCharge() * anInitialParticle.GetElectricPotential());
        double tNormalization = tFinalEnergy + tInitialEnergy;
        double tEnergyViolation = 2. * fabs( (tFinalEnergy / tNormalization) - (tInitialEnergy / tNormalization) )  ;

        if( tEnergyViolation < fLowerLimit )
        {
            trajmsg_debug( "stepsize energy increasing stepsize at violation <" << tEnergyViolation << ">" << eom) ;
            fTimeStep = (1. + fAdjustmentFactorUp) * fTimeStep;
            aFlag = false;
            return;
        }
        if( tEnergyViolation > fUpperLimit )
        {
            trajmsg_debug( "stepsize energy decreasing stepsize at violation <" << tEnergyViolation << ">" << eom );
            fTimeStep = (1. - fAdjustmentFactorDown) * fTimeStep;
            aFlag = false;
            return;
        }

        double tStepLength = aFinalParticle.GetLength() - anInitialParticle.GetLength();
        // particle's speed is calculated only when needed since GetVelocity() is a rather expensive operation

        if( tStepLength < fMinLength )
        {
            trajmsg_debug( "stepsize energy increasing stepsize at step length <" << tStepLength << ">" << eom) ;
            fTimeStep = (1. + fAdjustmentFactorUp) * fMinLength / anInitialParticle.GetVelocity().Magnitude();
            aFlag = false;
            return;
        }
        if( tStepLength > fMaxLength )
        {
            trajmsg_debug( "stepsize energy decreasing stepsize at step length <" << tStepLength << ">" << eom) ;
            fTimeStep = (1. - fAdjustmentFactorDown) * fMaxLength / anInitialParticle.GetVelocity().Magnitude();
            aFlag = false;
            return;
        }

        trajmsg_debug( "stepsize energy keeping stepsize at violation <" << tEnergyViolation << ">" << eom) ;
        return;
    }

}
