#include "KSRootSurfaceNavigator.h"

#include "KSNavigatorsMessage.h"

#include <limits>
using std::numeric_limits;

namespace Kassiopeia
{

    KSRootSurfaceNavigator::KSRootSurfaceNavigator() :
            fSurfaceNavigator( NULL ),
            fStep( NULL ),
            fTerminatorParticle( NULL ),
            fInteractionParticle( NULL ),
            fFinalParticle( NULL ),
            fParticleQueue( NULL )
    {
    }
    KSRootSurfaceNavigator::KSRootSurfaceNavigator( const KSRootSurfaceNavigator& aCopy ) :
            KSComponent(),
            fSurfaceNavigator( aCopy.fSurfaceNavigator ),
            fStep( aCopy.fStep ),
            fTerminatorParticle( aCopy.fTerminatorParticle ),
            fInteractionParticle( aCopy.fInteractionParticle ),
            fFinalParticle( aCopy.fFinalParticle ),
            fParticleQueue( aCopy.fParticleQueue )
    {
    }
    KSRootSurfaceNavigator* KSRootSurfaceNavigator::Clone() const
    {
        return new KSRootSurfaceNavigator( *this );
    }
    KSRootSurfaceNavigator::~KSRootSurfaceNavigator()
    {
    }

    void KSRootSurfaceNavigator::ExecuteNavigation( const KSParticle& anInitialParticle, const KSParticle& aNavigationParticle, KSParticle& aFinalParticle, KSParticleQueue& aSecondaries ) const
    {
        if( fSurfaceNavigator == NULL )
        {
            navmsg( eError ) << "<" << GetName() << "> cannot execute navigation with no surface navigation set" << eom;
        }
        fSurfaceNavigator->ExecuteNavigation( anInitialParticle, aNavigationParticle, aFinalParticle, aSecondaries );
        return;
    }

    void KSRootSurfaceNavigator::FinalizeNavigation( KSParticle& aFinalParticle ) const
    {
        if( fSurfaceNavigator == NULL )
        {
            navmsg( eError ) << "<" << GetName() << "> cannot finalize navigation with no surface navigation set" << eom;
        }
        fSurfaceNavigator->FinalizeNavigation( aFinalParticle );
        return;
    }

    void KSRootSurfaceNavigator::SetSurfaceNavigator( KSSurfaceNavigator* aSurfaceNavigator )
    {
        if( fSurfaceNavigator != NULL )
        {
            navmsg( eError ) << "<" << GetName() << "> tried to set surface navigator <" << aSurfaceNavigator->GetName() << "> with surface navigator <" << fSurfaceNavigator->GetName() << "> already set" << eom;
            return;
        }
        navmsg_debug( "<" << GetName() << "> setting surface navigator <" << aSurfaceNavigator->GetName() << ">" << eom );
        fSurfaceNavigator = aSurfaceNavigator;
        return;
    }
    void KSRootSurfaceNavigator::ClearSurfaceNavigator( KSSurfaceNavigator* aSurfaceNavigator )
    {
        if( fSurfaceNavigator != aSurfaceNavigator )
        {
            navmsg( eError ) << "<" << GetName() << "> tried to remove surface navigator <" << aSurfaceNavigator->GetName() << "> with surface navigator <" << fSurfaceNavigator->GetName() << "> already set" << eom;
            return;
        }
        navmsg_debug( "<" << GetName() << "> clearing surface navigator <" << aSurfaceNavigator->GetName() << ">" << eom );
        fSurfaceNavigator = NULL;
        return;
    }

    void KSRootSurfaceNavigator::SetStep( KSStep* aStep )
    {
        fStep = aStep;
        fTerminatorParticle = &(aStep->TerminatorParticle());
        fInteractionParticle = &(aStep->InteractionParticle());
        fFinalParticle = &(aStep->FinalParticle());
        fParticleQueue = &(aStep->ParticleQueue());
        return;
    }

    void KSRootSurfaceNavigator::ExecuteNavigation()
    {
        ExecuteNavigation(  *fTerminatorParticle, *fInteractionParticle, *fFinalParticle, *fParticleQueue );
        fStep->SurfaceNavigationFlag() = true;
        fFinalParticle->ReleaseLabel( fStep->SurfaceNavigationName() );

        fStep->ContinuousTime() = 0.;
        fStep->ContinuousLength() = 0.;
        fStep->ContinuousEnergyChange() = 0.;
        fStep->ContinuousMomentumChange() = 0.;
        fStep->DiscreteSecondaries() += fParticleQueue->size();
        fStep->DiscreteEnergyChange() += fFinalParticle->GetKineticEnergy_eV() - fInteractionParticle->GetKineticEnergy_eV();
        fStep->DiscreteMomentumChange() += (fFinalParticle->GetMomentum() - fInteractionParticle->GetMomentum()).Magnitude();

        navmsg_debug( "surface navigation:" << eom );
        navmsg_debug( "  surface navigation name: <" << fStep->GetSurfaceNavigationName() << ">" << eom );
        navmsg_debug( "  step continuous time: <" << fStep->ContinuousTime() << ">" << eom );
        navmsg_debug( "  step continuous length: <" << fStep->ContinuousLength() << ">" << eom );
        navmsg_debug( "  step continuous energy change: <" << fStep->ContinuousEnergyChange() << ">" << eom );
        navmsg_debug( "  step continuous momentum change: <" << fStep->ContinuousMomentumChange() << ">" << eom );
        navmsg_debug( "  step discrete secondaries: <" << fStep->DiscreteSecondaries() << ">" << eom );
        navmsg_debug( "  step discrete energy change: <" << fStep->DiscreteEnergyChange() << ">" << eom );
        navmsg_debug( "  step discrete momentum change: <" << fStep->DiscreteMomentumChange() << ">" << eom );

        navmsg_debug( "surface navigation final particle state: " << eom );
        navmsg_debug( "  final particle space: <" << (fFinalParticle->GetCurrentSpace() ? fFinalParticle->GetCurrentSpace()->GetName() : "" ) << ">" << eom )
        navmsg_debug( "  final particle surface: <" << (fFinalParticle->GetCurrentSurface() ? fFinalParticle->GetCurrentSurface()->GetName() : "") << ">" << eom );
        navmsg_debug( "  final particle side: <" << (fFinalParticle->GetCurrentSide() ? fFinalParticle->GetCurrentSide()->GetName() : "") << ">" << eom );
        navmsg_debug( "  final particle time: <" << fFinalParticle->GetTime() << ">" << eom );
        navmsg_debug( "  final particle length: <" << fFinalParticle->GetLength() << ">" << eom );
        navmsg_debug( "  final particle position: <" << fFinalParticle->GetPosition().X() << ", " << fFinalParticle->GetPosition().Y() << ", " << fFinalParticle->GetPosition().Z() << ">" << eom );
        navmsg_debug( "  final particle momentum: <" << fFinalParticle->GetMomentum().X() << ", " << fFinalParticle->GetMomentum().Y() << ", " << fFinalParticle->GetMomentum().Z() << ">" << eom );
        navmsg_debug( "  final particle kinetic energy: <" << fFinalParticle->GetKineticEnergy_eV() << ">" << eom );
        navmsg_debug( "  final particle electric field: <" << fFinalParticle->GetElectricField().X() << "," << fFinalParticle->GetElectricField().Y() << "," << fFinalParticle->GetElectricField().Z() << ">" << eom );
        navmsg_debug( "  final particle magnetic field: <" << fFinalParticle->GetMagneticField().X() << "," << fFinalParticle->GetMagneticField().Y() << "," << fFinalParticle->GetMagneticField().Z() << ">" << eom );
        navmsg_debug( "  final particle angle to magnetic field: <" << fFinalParticle->GetPolarAngleToB() << ">" << eom );

        return;
    }

    void KSRootSurfaceNavigator::FinalizeNavigation()
	{
    	FinalizeNavigation( *fFinalParticle );
		return;
	}

    STATICINT sKSRootSurfaceNavigatorDict =
        KSDictionary< KSRootSurfaceNavigator >::AddCommand( &KSRootSurfaceNavigator::SetSurfaceNavigator, &KSRootSurfaceNavigator::ClearSurfaceNavigator, "set_surface_navigator", "clear_surface_navigator" );

}
