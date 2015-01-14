#include "KSRootTrajectory.h"
#include "KSTrajectoriesMessage.h"

namespace Kassiopeia
{

    KSRootTrajectory::KSRootTrajectory() :
            fTrajectory( NULL ),
            fStep( NULL ),
            fTerminatorParticle( NULL ),
            fTrajectoryParticle( NULL ),
            fFinalParticle( NULL )
    {
    }
    KSRootTrajectory::KSRootTrajectory( const KSRootTrajectory& aCopy ) :
            fTrajectory( aCopy.fTrajectory ),
            fStep( aCopy.fStep ),
            fTerminatorParticle( aCopy.fTerminatorParticle ),
            fTrajectoryParticle( aCopy.fTrajectoryParticle ),
            fFinalParticle( aCopy.fFinalParticle )
    {
    }
    KSRootTrajectory* KSRootTrajectory::Clone() const
    {
        return new KSRootTrajectory( *this );
    }
    KSRootTrajectory::~KSRootTrajectory()
    {
    }

    void KSRootTrajectory::CalculateTrajectory( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KThreeVector& aCenter, double& aRadius, double& aTimeStep )
    {
        if( fTrajectory == NULL )
        {
            trajmsg( eError ) << "<" << GetName() << "> cannot calculate trajectory with no trajectory set" << eom;
        }
        fTrajectory->CalculateTrajectory( anInitialParticle, aFinalParticle, aCenter, aRadius, aTimeStep );
        return;
    }
    void KSRootTrajectory::ExecuteTrajectory( const double& aTimeStep, KSParticle& anIntermediateParticle ) const
    {
        if( fTrajectory == NULL )
        {
            trajmsg( eError ) << "<" << GetName() << "> cannot execute trajectory with no trajectory set" << eom;
        }
        fTrajectory->ExecuteTrajectory( aTimeStep, anIntermediateParticle );
        return;
    }

    void KSRootTrajectory::SetTrajectory( KSTrajectory* aTrajectory )
    {
        if( fTrajectory != NULL )
        {
            trajmsg( eError ) << "<" << GetName() << "> tried to set trajectory <" << aTrajectory->GetName() << "> with trajectory <" << fTrajectory->GetName() << "> already set" << eom;
            return;
        }
        trajmsg_debug( "<" << GetName() << "> setting trajectory <" << aTrajectory->GetName() << ">" << eom );
        fTrajectory = aTrajectory;
        return;
    }
    void KSRootTrajectory::ClearTrajectory( KSTrajectory* aTrajectory )
    {
        if( fTrajectory != aTrajectory )
        {
            trajmsg( eError ) << "<" << GetName() << "> tried to remove trajectory <" << aTrajectory->GetName() << "> with trajectory <" << fTrajectory->GetName() << "> already set" << eom;
            return;
        }
        trajmsg_debug( "<" << GetName() << "> clearing trajectory <" << aTrajectory->GetName() << ">" << eom );
        fTrajectory = NULL;
        return;
    }

    void KSRootTrajectory::SetStep( KSStep* aStep )
    {
        fStep = aStep;
        fTerminatorParticle = &(aStep->TerminatorParticle());
        fTrajectoryParticle = &(aStep->TrajectoryParticle());
        fFinalParticle = &(aStep->FinalParticle());
        return;
    }

    void KSRootTrajectory::CalculateTrajectory()
    {
        *fTrajectoryParticle = *fTerminatorParticle;

        CalculateTrajectory( *fTerminatorParticle, *fTrajectoryParticle, fStep->TrajectoryCenter(), fStep->TrajectoryRadius(), fStep->TrajectoryStep() );
        fFinalParticle->ReleaseLabel( fStep->TrajectoryName() );

        trajmsg_debug( "trajectory calculation:" << eom );
        trajmsg_debug( "  trajectory name: <" << fStep->TrajectoryName() << ">" << eom );
        trajmsg_debug( "  trajectory center: <" << fStep->TrajectoryCenter().X() << ", " << fStep->TrajectoryCenter().Y() << ", " << fStep->TrajectoryCenter().Z() << ">" << eom );
        trajmsg_debug( "  trajectory radius: <" << fStep->TrajectoryRadius() << ">" << eom );
        trajmsg_debug( "  trajectory step: <" << fStep->TrajectoryStep() << ">" << eom );

        trajmsg_debug( "trajectory calculation trajectory particle state: " << eom );
        trajmsg_debug( "  trajectory particle space: <" << (fTrajectoryParticle->GetCurrentSpace() ? fTrajectoryParticle->GetCurrentSpace()->GetName() : "" ) << ">" << eom );
        trajmsg_debug( "  trajectory particle surface: <" << (fTrajectoryParticle->GetCurrentSurface() ? fTrajectoryParticle->GetCurrentSurface()->GetName() : "" ) << ">" << eom );
        trajmsg_debug( "  trajectory particle time: <" << fTrajectoryParticle->GetTime() << ">" << eom );
        trajmsg_debug( "  trajectory particle length: <" << fTrajectoryParticle->GetLength() << ">" << eom );
        trajmsg_debug( "  trajectory particle position: <" << fTrajectoryParticle->GetPosition().X() << ", " << fTrajectoryParticle->GetPosition().Y() << ", " << fTrajectoryParticle->GetPosition().Z() << ">" << eom );
        trajmsg_debug( "  trajectory particle momentum: <" << fTrajectoryParticle->GetMomentum().X() << ", " << fTrajectoryParticle->GetMomentum().Y() << ", " << fTrajectoryParticle->GetMomentum().Z() << ">" << eom );
        trajmsg_debug( "  trajectory particle kinetic energy: <" << fTrajectoryParticle->GetKineticEnergy_eV() << ">" << eom );
        trajmsg_debug( "  trajectory particle electric field: <" << fTrajectoryParticle->GetElectricField().X() << "," << fTrajectoryParticle->GetElectricField().Y() << "," << fTrajectoryParticle->GetElectricField().Z() << ">" << eom );
        trajmsg_debug( "  trajectory particle magnetic field: <" << fTrajectoryParticle->GetMagneticField().X() << "," << fTrajectoryParticle->GetMagneticField().Y() << "," << fTrajectoryParticle->GetMagneticField().Z() << ">" << eom );
        trajmsg_debug( "  trajectory particle angle to magnetic field: <" << fTrajectoryParticle->GetPolarAngleToB() << ">" << eom );

        return;
    }

    void KSRootTrajectory::ExecuteTrajectory()
    {
        *fFinalParticle = *fTrajectoryParticle;

        fStep->ContinuousTime() = fFinalParticle->GetTime() - fTerminatorParticle->GetTime();
        fStep->ContinuousLength() = fFinalParticle->GetLength() - fTerminatorParticle->GetLength();
        fStep->ContinuousEnergyChange() = fFinalParticle->GetKineticEnergy_eV() - fTerminatorParticle->GetKineticEnergy_eV();
        fStep->ContinuousMomentumChange() = (fFinalParticle->GetMomentum() - fTerminatorParticle->GetMomentum()).Magnitude();
        fStep->DiscreteSecondaries() = 0;
        fStep->DiscreteEnergyChange() = 0.;
        fStep->DiscreteMomentumChange() = 0.;

        trajmsg_debug( "trajectory execution:" << eom );
        trajmsg_debug( "  step continuous time: <" << fStep->ContinuousTime() << ">" << eom );
        trajmsg_debug( "  step continuous length: <" << fStep->ContinuousLength() << ">" << eom );
        trajmsg_debug( "  step continuous energy change: <" << fStep->ContinuousEnergyChange() << ">" << eom );
        trajmsg_debug( "  step continuous momentum change: <" << fStep->ContinuousMomentumChange() << ">" << eom );
        trajmsg_debug( "  step discrete secondaries: <" << fStep->DiscreteSecondaries() << ">" << eom );
        trajmsg_debug( "  step discrete energy change: <" << fStep->DiscreteEnergyChange() << ">" << eom );
        trajmsg_debug( "  step discrete momentum change: <" << fStep->DiscreteMomentumChange() << ">" << eom );

        trajmsg_debug( "trajectory execution final particle state: " << eom );
        trajmsg_debug( "  final particle space: <" << (fFinalParticle->GetCurrentSpace() ? fFinalParticle->GetCurrentSpace()->GetName() : "" ) << ">" << eom );
        trajmsg_debug( "  final particle surface: <" << (fFinalParticle->GetCurrentSurface() ? fFinalParticle->GetCurrentSurface()->GetName() : "" ) << ">" << eom );
        trajmsg_debug( "  final particle time: <" << fFinalParticle->GetTime() << ">" << eom );
        trajmsg_debug( "  final particle length: <" << fFinalParticle->GetLength() << ">" << eom );
        trajmsg_debug( "  final particle position: <" << fFinalParticle->GetPosition().X() << ", " << fFinalParticle->GetPosition().Y() << ", " << fFinalParticle->GetPosition().Z() << ">" << eom );
        trajmsg_debug( "  final particle momentum: <" << fFinalParticle->GetMomentum().X() << ", " << fFinalParticle->GetMomentum().Y() << ", " << fFinalParticle->GetMomentum().Z() << ">" << eom );
        trajmsg_debug( "  final particle kinetic energy: <" << fFinalParticle->GetKineticEnergy_eV() << ">" << eom );
        trajmsg_debug( "  final particle electric field: <" << fFinalParticle->GetElectricField().X() << "," << fFinalParticle->GetElectricField().Y() << "," << fFinalParticle->GetElectricField().Z() << ">" << eom );
        trajmsg_debug( "  final particle magnetic field: <" << fFinalParticle->GetMagneticField().X() << "," << fFinalParticle->GetMagneticField().Y() << "," << fFinalParticle->GetMagneticField().Z() << ">" << eom );
        trajmsg_debug( "  final particle angle to magnetic field: <" << fFinalParticle->GetPolarAngleToB() << ">" << eom );

        return;
    }

    static const int sKSRootTrajectoryDict =
        KSDictionary< KSRootTrajectory >::AddCommand( &KSRootTrajectory::SetTrajectory, &KSRootTrajectory::ClearTrajectory, "set_trajectory", "clear_trajectory" );

}
