#include "KSIntSpinRotateYPulse.h"
#include "KSInteractionsMessage.h"

#include "KRandom.h"
#include "KConst.h"
using katrin::KRandom;

namespace Kassiopeia
{

    KSIntSpinRotateYPulse::KSIntSpinRotateYPulse() :
            fDone( false ),
            fTime( 0. ),
            fAngle( 0. ),
            fIsAdiabatic( true ) // defaulting to true since I expect that to be the more common use case
    {
    }

    KSIntSpinRotateYPulse::KSIntSpinRotateYPulse( const KSIntSpinRotateYPulse& aCopy ) :
            KSComponent(),
            KSComponentTemplate< KSIntSpinRotateYPulse, KSSpaceInteraction >( aCopy ),
            fDone( aCopy.fDone ),
            fTime ( aCopy.fTime ),
            fAngle ( aCopy.fAngle ),
            fIsAdiabatic ( aCopy.fIsAdiabatic )
    {
    }

    KSIntSpinRotateYPulse* KSIntSpinRotateYPulse::Clone() const
    {
        return new KSIntSpinRotateYPulse( *this );
    }

    KSIntSpinRotateYPulse::~KSIntSpinRotateYPulse()
    {
    }


    void KSIntSpinRotateYPulse::CalculateInteraction(
            const KSTrajectory& /*aTrajectory*/,
            const KSParticle& aTrajectoryInitialParticle,
            const KSParticle& aTrajectoryFinalParticle,
            const KThreeVector& /*aTrajectoryCenter*/,
            const double& /*aTrajectoryRadius*/,
            const double& /*aTrajectoryTimeStep*/,
            KSParticle& anInteractionParticle,
            double& aTimeStep,
            bool& aFlag
            )
    {
        anInteractionParticle = aTrajectoryFinalParticle;

        if ( aTrajectoryFinalParticle.GetTime() < fTime )
        {
            aFlag = false;
            fDone = false;
        }
        else if ( fDone )
        {
            aFlag = false;
        }
        else
        {
            aFlag = true;
            //fDone = true;  //Due to double precision errors in the time, this should occur on execution instead
            aTimeStep = 0.999 * ( fTime - aTrajectoryInitialParticle.GetTime() );
        }

        return;
    }

    void KSIntSpinRotateYPulse::ExecuteInteraction( const KSParticle& anInteractionParticle,
                                         KSParticle& aFinalParticle,
                                         KSParticleQueue& /*aSecondaries*/ ) const
    {

        aFinalParticle = anInteractionParticle;

        if (fDone){ // This is an awful hack, but is necessary for some reason
            return;
        }
        fDone = true;

        if (fIsAdiabatic){
            aFinalParticle.RecalculateSpinGlobal();
        }

        double x = aFinalParticle.GetSpinX();
        double z = aFinalParticle.GetSpinZ();
        aFinalParticle.SetSpinX( x*cos( KConst::Pi() / 180 * fAngle ) + z*sin( KConst::Pi() / 180 * fAngle ) );
        aFinalParticle.SetSpinZ( z*cos( KConst::Pi() / 180 * fAngle ) - x*sin( KConst::Pi() / 180 * fAngle ) );

        aFinalParticle.RecalculateSpinBody();

        return;
    }

    void KSIntSpinRotateYPulse::SetTime( const double& aTime )
    {
        fTime = aTime;
        return;
    }

    void KSIntSpinRotateYPulse::SetAngle( const double& anAngle )
    {
        fAngle = anAngle;
        return;
    }

    void KSIntSpinRotateYPulse::SetIsAdiabatic( const bool& anIsAdiabatic )
    {
        fIsAdiabatic = anIsAdiabatic;
        return;
    }

}
