    #include "KSTrajControlEnergy.h"
#include "KSTrajectoriesMessage.h"

namespace Kassiopeia
{

    KSTrajControlEnergy::KSTrajControlEnergy() :
            fLowerLimit( 1.e-12 ),
            fUpperLimit( 1.e-10 ),
            fTimeStep( 0. ),
            fFirstStep( true )
    {
    }
    KSTrajControlEnergy::KSTrajControlEnergy( const KSTrajControlEnergy& aCopy ) :
            fLowerLimit( aCopy.fLowerLimit ),
            fUpperLimit( aCopy.fUpperLimit ),
            fTimeStep( aCopy.fTimeStep ),
            fFirstStep( true )
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
        trajmsg_debug( "stepsize energy resetting" << eom );
        fFirstStep = true;
        return;
    }

    void KSTrajControlEnergy::Calculate( const KSTrajExactParticle& aParticle, double& aValue )
    {
        if( fFirstStep == true )
        {
            trajmsg_debug( "stepsize energy on first step" << eom );
            fTimeStep = 0.0625 / aParticle.GetCyclotronFrequency();
            fFirstStep = false;
        }

        trajmsg_debug( "stepsize energy suggesting <" << fTimeStep << ">" << eom );
        aValue = fTimeStep;
        return;
    }
    void KSTrajControlEnergy::Check( const KSTrajExactParticle& anInitialParticle, const KSTrajExactParticle& aFinalParticle, const KSTrajExactError&, bool& aFlag )
    {
        double tFinalEnergy = (aFinalParticle.GetKineticEnergy() + aFinalParticle.GetCharge() * aFinalParticle.GetElectricPotential());
        double tInitialEnergy = (anInitialParticle.GetKineticEnergy() + anInitialParticle.GetCharge() * anInitialParticle.GetElectricPotential());
        double tEnergyViolation = fabs( 2. * (tFinalEnergy - tInitialEnergy) / (tFinalEnergy + tInitialEnergy) );

        fTimeStep = aFinalParticle.GetTime() - anInitialParticle.GetTime();

        if( tEnergyViolation < fLowerLimit )
        {
            trajmsg_debug( "stepsize energy increasing stepsize at violation <" << tEnergyViolation << ">" << eom) ;
            fTimeStep = 1.5 * fTimeStep;
            aFlag = true;
            return;
        }

        if( tEnergyViolation > fUpperLimit )
        {
            trajmsg_debug( "stepsize energy decreasing stepsize at violation <" << tEnergyViolation << ">" << eom );
            fTimeStep = 0.4 * fTimeStep;
            aFlag = false;
            return;
        }

        trajmsg_debug( "stepsize energy keeping stepsize at violation <" << tEnergyViolation << ">" << eom) ;
        aFlag = true;
        return;
    }

    void KSTrajControlEnergy::Calculate( const KSTrajAdiabaticParticle& aParticle, double& aValue )
    {
        if( fFirstStep == true )
        {
            trajmsg_debug( "stepsize energy on first step" << eom );
            fTimeStep = 0.0625 / aParticle.GetCyclotronFrequency();
            fFirstStep = false;
        }

        trajmsg_debug( "stepsize energy suggesting <" << fTimeStep << ">" << eom );
        aValue = fTimeStep;
        return;
    }
    void KSTrajControlEnergy::Check( const KSTrajAdiabaticParticle& anInitialParticle, const KSTrajAdiabaticParticle& aFinalParticle, const KSTrajAdiabaticError&, bool& aFlag )
    {
        double tFinalEnergy = (aFinalParticle.GetKineticEnergy() + aFinalParticle.GetCharge() * aFinalParticle.GetElectricPotential());
        double tInitialEnergy = (anInitialParticle.GetKineticEnergy() + anInitialParticle.GetCharge() * anInitialParticle.GetElectricPotential());
        double tEnergyViolation = fabs( 2. * (tFinalEnergy - tInitialEnergy) / (tFinalEnergy + tInitialEnergy) );

        fTimeStep = aFinalParticle.GetTime() - anInitialParticle.GetTime();

        if( tEnergyViolation < fLowerLimit )
        {
            trajmsg_debug( "stepsize energy increasing stepsize at violation <" << tEnergyViolation << ">" << eom) ;
            fTimeStep = 1.5 * fTimeStep;
            aFlag = true;
            return;
        }

        if( tEnergyViolation > fUpperLimit )
        {
            trajmsg_debug( "stepsize energy decreasing stepsize at violation <" << tEnergyViolation << ">" << eom );
            fTimeStep = 0.4 * fTimeStep;
            aFlag = false;
            return;
        }

        trajmsg_debug( "stepsize energy keeping stepsize at violation <" << tEnergyViolation << ">" << eom) ;
        aFlag = true;
        return;
    }

}
