#include "KSRootTerminator.h"
#include "KSTerminatorsMessage.h"

namespace Kassiopeia
{

    KSRootTerminator::KSRootTerminator() :
            fTerminators( 128 ),
            fTerminator( NULL ),
            fStep( NULL ),
            fInitialParticle( NULL ),
            fTerminatorParticle( NULL ),
            fFinalParticle( NULL ),
            fParticleQueue( NULL )
    {
    }
    KSRootTerminator::KSRootTerminator( const KSRootTerminator& aCopy ) :
            KSComponent(),
            fTerminators( aCopy.fTerminators ),
            fTerminator( aCopy.fTerminator ),
            fStep( aCopy.fStep ),
            fInitialParticle( aCopy.fInitialParticle ),
            fTerminatorParticle( aCopy.fTerminatorParticle ),
            fFinalParticle( aCopy.fFinalParticle ),
            fParticleQueue( aCopy.fParticleQueue )
    {
    }
    KSRootTerminator* KSRootTerminator::Clone() const
    {
        return new KSRootTerminator( *this );
    }
    KSRootTerminator::~KSRootTerminator()
    {
    }

    void KSRootTerminator::CalculateTermination( const KSParticle& anInitialParticle, bool& aFlag )
    {
        bool tTerminatorFlag;

        aFlag = false;
        fTerminator = NULL;
        for( int tIndex = 0; tIndex < fTerminators.End(); tIndex++ )
        {
            fTerminators.ElementAt( tIndex )->CalculateTermination( anInitialParticle, tTerminatorFlag );
            if( tTerminatorFlag == true )
            {
                aFlag = true;
                fTerminator = fTerminators.ElementAt( tIndex );
                return;
            }
        }

        return;
    }
    void KSRootTerminator::ExecuteTermination( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aParticleQueue ) const
    {
        if( fTerminator != NULL )
        {
            fTerminator->ExecuteTermination( anInitialParticle, aFinalParticle, aParticleQueue );
        }
        else
        {
            aFinalParticle = anInitialParticle;
        }
        return;
    }

    void KSRootTerminator::AddTerminator( KSTerminator* aTerminator )
    {
        fTerminators.AddElement( aTerminator );
        return;
    }
    void KSRootTerminator::RemoveTerminator( KSTerminator* aTerminator )
    {
        fTerminators.RemoveElement( aTerminator );
        return;
    }

    void KSRootTerminator::SetStep( KSStep* aStep )
    {
        fStep = aStep;
        fInitialParticle = &(aStep->InitialParticle());
        fTerminatorParticle = &(aStep->TerminatorParticle());
        fFinalParticle = &(aStep->FinalParticle());
        fParticleQueue = &(aStep->ParticleQueue());
        return;
    }

    void KSRootTerminator::PushUpdateComponent()
    {
        for( int tIndex = 0; tIndex < fTerminators.End(); tIndex++ )
        {
            fTerminators.ElementAt( tIndex )->PushUpdate();
        }
    }

    void KSRootTerminator::PushDeupdateComponent()
    {
        for( int tIndex = 0; tIndex < fTerminators.End(); tIndex++ )
        {
            fTerminators.ElementAt( tIndex )->PushDeupdate();
        }
    }

    void KSRootTerminator::CalculateTermination()
    {
        *fTerminatorParticle = *fInitialParticle;
        fStep->TerminatorName().clear();
        fStep->TerminatorFlag() = false;

        if( fTerminators.End() == 0 )
        {
            termmsg_debug( "terminator calculation:" << eom )
            termmsg_debug( "  no terminators active" << eom )
            termmsg_debug( "  terminator name: <" << fStep->GetTerminatorName() << ">" << eom )
            termmsg_debug( "  terminator flag: <" << fStep->GetTerminatorFlag() << ">" << eom )

            termmsg_debug( "terminator calculation terminator particle state: " << eom )
            termmsg_debug( "  terminator particle space: <" << (fTerminatorParticle->GetCurrentSpace() ? fTerminatorParticle->GetCurrentSpace()->GetName() : "" ) << ">" << eom )
            termmsg_debug( "  terminator particle surface: <" << (fTerminatorParticle->GetCurrentSurface() ? fTerminatorParticle->GetCurrentSurface()->GetName() : "" ) << ">" << eom )
            termmsg_debug( "  terminator particle time: <" << fTerminatorParticle->GetTime() << ">" << eom )
            termmsg_debug( "  terminator particle length: <" << fTerminatorParticle->GetLength() << ">" << eom )
            termmsg_debug( "  terminator particle position: <" << fTerminatorParticle->GetPosition().X() << ", " << fTerminatorParticle->GetPosition().Y() << ", " << fTerminatorParticle->GetPosition().Z() << ">" << eom )
            termmsg_debug( "  terminator particle momentum: <" << fTerminatorParticle->GetMomentum().X() << ", " << fTerminatorParticle->GetMomentum().Y() << ", " << fTerminatorParticle->GetMomentum().Z() << ">" << eom )
            termmsg_debug( "  terminator particle kinetic energy: <" << fTerminatorParticle->GetKineticEnergy_eV() << ">" << eom )
            termmsg_debug( "  terminator particle electric field: <" << fTerminatorParticle->GetElectricField().X() << "," << fTerminatorParticle->GetElectricField().Y() << "," << fTerminatorParticle->GetElectricField().Z() << ">" << eom )
            termmsg_debug( "  terminator particle magnetic field: <" << fTerminatorParticle->GetMagneticField().X() << "," << fTerminatorParticle->GetMagneticField().Y() << "," << fTerminatorParticle->GetMagneticField().Z() << ">" << eom )
            termmsg_debug( "  terminator particle angle to magnetic field: <" << fTerminatorParticle->GetPolarAngleToB() << ">" << eom )
            termmsg_debug( "  terminator particle spin: " << fTerminatorParticle->GetSpin() << eom )
            termmsg_debug( "  terminator particle spin0: <" << fTerminatorParticle->GetSpin0() << ">" << eom )
            termmsg_debug( "  terminator particle aligned spin: <" << fTerminatorParticle->GetAlignedSpin() << ">" << eom )
            termmsg_debug( "  terminator particle spin angle: <" << fTerminatorParticle->GetSpinAngle() << ">" << eom )

            return;
        }

        CalculateTermination( *fTerminatorParticle, fStep->TerminatorFlag() );

        if( fStep->TerminatorFlag() == true )
        {
            termmsg_debug( "terminator calculation:" << eom )
            termmsg_debug( "  terminator may occur" << eom )
        }
        else
        {
            termmsg_debug( "terminator calculation:" << eom )
            termmsg_debug( "  terminator will not occur" << eom )
        }

        termmsg_debug( "terminator calculation terminator particle state: " << eom )
        termmsg_debug( "  terminator particle space: <" << (fTerminatorParticle->GetCurrentSpace() ? fTerminatorParticle->GetCurrentSpace()->GetName() : "" ) << ">" << eom )
        termmsg_debug( "  terminator particle surface: <" << (fTerminatorParticle->GetCurrentSurface() ? fTerminatorParticle->GetCurrentSurface()->GetName() : "" ) << ">" << eom )
        termmsg_debug( "  terminator particle time: <" << fTerminatorParticle->GetTime() << ">" << eom )
        termmsg_debug( "  terminator particle length: <" << fTerminatorParticle->GetLength() << ">" << eom )
        termmsg_debug( "  terminator particle position: <" << fTerminatorParticle->GetPosition().X() << ", " << fTerminatorParticle->GetPosition().Y() << ", " << fTerminatorParticle->GetPosition().Z() << ">" << eom )
        termmsg_debug( "  terminator particle momentum: <" << fTerminatorParticle->GetMomentum().X() << ", " << fTerminatorParticle->GetMomentum().Y() << ", " << fTerminatorParticle->GetMomentum().Z() << ">" << eom )
        termmsg_debug( "  terminator particle kinetic energy: <" << fTerminatorParticle->GetKineticEnergy_eV() << ">" << eom )
        termmsg_debug( "  terminator particle electric field: <" << fTerminatorParticle->GetElectricField().X() << "," << fTerminatorParticle->GetElectricField().Y() << "," << fTerminatorParticle->GetElectricField().Z() << ">" << eom )
        termmsg_debug( "  terminator particle magnetic field: <" << fTerminatorParticle->GetMagneticField().X() << "," << fTerminatorParticle->GetMagneticField().Y() << "," << fTerminatorParticle->GetMagneticField().Z() << ">" << eom )
        termmsg_debug( "  terminator particle angle to magnetic field: <" << fTerminatorParticle->GetPolarAngleToB() << ">" << eom )
        termmsg_debug( "  terminator particle spin: " << fTerminatorParticle->GetSpin() << eom )
        termmsg_debug( "  terminator particle spin0: <" << fTerminatorParticle->GetSpin0() << ">" << eom )
        termmsg_debug( "  terminator particle aligned spin: <" << fTerminatorParticle->GetAlignedSpin() << ">" << eom )
        termmsg_debug( "  terminator particle spin angle: <" << fTerminatorParticle->GetSpinAngle() << ">" << eom )

        return;
    }

    void KSRootTerminator::ExecuteTermination()
    {
        ExecuteTermination( *fTerminatorParticle, *fFinalParticle, *fParticleQueue );
        fFinalParticle->ReleaseLabel( fStep->TerminatorName() );

        fStep->ContinuousTime() = 0.;
        fStep->ContinuousLength() = 0.;
        fStep->ContinuousEnergyChange() = 0.;
        fStep->ContinuousMomentumChange() = 0.;
        fStep->DiscreteSecondaries() = 0;
        fStep->DiscreteEnergyChange() = 0.;
        fStep->DiscreteMomentumChange() = 0.;

        termmsg_debug( "terminator execution:" << eom )
        termmsg_debug( "  terminator name: <" << fStep->TerminatorName() << ">" << eom )
        termmsg_debug( "  step continuous time: <" << fStep->ContinuousTime() << ">" << eom )
        termmsg_debug( "  step continuous length: <" << fStep->ContinuousLength() << ">" << eom )
        termmsg_debug( "  step continuous energy change: <" << fStep->ContinuousEnergyChange() << ">" << eom )
        termmsg_debug( "  step continuous momentum change: <" << fStep->ContinuousMomentumChange() << ">" << eom )
        termmsg_debug( "  step discrete secondaries: <" << fStep->DiscreteSecondaries() << ">" << eom )
        termmsg_debug( "  step discrete energy change: <" << fStep->DiscreteEnergyChange() << ">" << eom )
        termmsg_debug( "  step discrete momentum change: <" << fStep->DiscreteMomentumChange() << ">" << eom )

        termmsg_debug( "terminator final particle state: " << eom )
        termmsg_debug( "  final particle space: <" << (fTerminatorParticle->GetCurrentSpace() ? fTerminatorParticle->GetCurrentSpace()->GetName() : "" ) << ">" << eom )
        termmsg_debug( "  final particle surface: <" << (fTerminatorParticle->GetCurrentSurface() ? fTerminatorParticle->GetCurrentSurface()->GetName() : "" ) << ">" << eom )
        termmsg_debug( "  final particle time: <" << fTerminatorParticle->GetTime() << ">" << eom )
        termmsg_debug( "  final particle length: <" << fTerminatorParticle->GetLength() << ">" << eom )
        termmsg_debug( "  final particle position: <" << fTerminatorParticle->GetPosition().X() << ", " << fTerminatorParticle->GetPosition().Y() << ", " << fTerminatorParticle->GetPosition().Z() << ">" << eom )
        termmsg_debug( "  final particle momentum: <" << fTerminatorParticle->GetMomentum().X() << ", " << fTerminatorParticle->GetMomentum().Y() << ", " << fTerminatorParticle->GetMomentum().Z() << ">" << eom )
        termmsg_debug( "  final particle kinetic energy: <" << fTerminatorParticle->GetKineticEnergy_eV() << ">" << eom )
        termmsg_debug( "  final particle electric field: <" << fTerminatorParticle->GetElectricField().X() << "," << fTerminatorParticle->GetElectricField().Y() << "," << fTerminatorParticle->GetElectricField().Z() << ">" << eom )
        termmsg_debug( "  final particle magnetic field: <" << fTerminatorParticle->GetMagneticField().X() << "," << fTerminatorParticle->GetMagneticField().Y() << "," << fTerminatorParticle->GetMagneticField().Z() << ">" << eom )
        termmsg_debug( "  final particle angle to magnetic field: <" << fTerminatorParticle->GetPolarAngleToB() << ">" << eom )
        termmsg_debug( "  final particle spin: " << fTerminatorParticle->GetSpin() << eom )
        termmsg_debug( "  final particle spin0: <" << fTerminatorParticle->GetSpin0() << ">" << eom )
        termmsg_debug( "  final particle aligned spin: <" << fTerminatorParticle->GetAlignedSpin() << ">" << eom )
        termmsg_debug( "  final particle spin angle: <" << fTerminatorParticle->GetSpinAngle() << ">" << eom )

        return;
    }

    STATICINT sKSRootTerminatorDict =
        KSDictionary< KSRootTerminator >::AddCommand( &KSRootTerminator::AddTerminator, &KSRootTerminator::RemoveTerminator, "add_terminator", "remove_terminator" );

}
