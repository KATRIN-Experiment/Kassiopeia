#include "KSRootStepModifier.h"
#include "KSModifiersMessage.h"

namespace Kassiopeia
{
    KSRootStepModifier::KSRootStepModifier() :
        fModifiers(128),
        fModifier( NULL ),
        fInitialParticle( NULL ),
        fFinalParticle( NULL ),
        fParticleQueue( NULL )
    {
    }

    KSRootStepModifier::KSRootStepModifier(const KSRootStepModifier &aCopy) : KSComponent(),
        fModifiers( aCopy.fModifiers ),
        fModifier( aCopy.fModifier),
        fInitialParticle( aCopy.fInitialParticle ),
        fFinalParticle( aCopy.fFinalParticle ),
        fParticleQueue( aCopy.fParticleQueue )
    {
    }
    KSRootStepModifier* KSRootStepModifier::Clone() const
    {
        return new KSRootStepModifier( *this );
    }
    KSRootStepModifier::~KSRootStepModifier()
    {
    }

    void KSRootStepModifier::AddModifier(KSStepModifier *aModifier)
    {
        fModifiers.AddElement( aModifier );
        return;
    }
    void KSRootStepModifier::RemoveModifier(KSStepModifier *aModifier)
    {
        fModifiers.RemoveElement( aModifier );
        return;
    }
    void KSRootStepModifier::SetStep( KSStep* aStep )
    {
        fStep = aStep;
        fInitialParticle = &(aStep->InitialParticle());
        fModifierParticle = &(aStep->TerminatorParticle());
        fFinalParticle = &(aStep->FinalParticle());
        fParticleQueue = &(aStep->ParticleQueue());

        return;
    }

    void KSRootStepModifier::PushUpdateComponent()
    {
        for( int tIndex = 0; tIndex < fModifiers.End(); tIndex++ )
        {
            fModifiers.ElementAt( tIndex )->PushUpdate();
        }
    }

    void KSRootStepModifier::PushDeupdateComponent()
    {
        for( int tIndex = 0; tIndex < fModifiers.End(); tIndex++ )
        {
            fModifiers.ElementAt( tIndex )->PushDeupdate();
        }
    }

    bool KSRootStepModifier::ExecutePreStepModification()
    {
        //the following disables any change made to the initial particle, why?
        *fModifierParticle = *fInitialParticle;
        fStep->ModifierName().clear();
        fStep->ModifierFlag() = false;

        if( fModifiers.End() == 0 )
        {
            modmsg_debug( "modifier calculation:" << eom )
            modmsg_debug( "  no modifier active" << eom )
            modmsg_debug( "  modifier name: <" << fStep->GetModifierName() << ">" << eom )
            modmsg_debug( "  modifier flag: <" << fStep->GetModifierFlag() << ">" << eom )

            modmsg_debug( "modifier calculation terminator particle state: " << eom )
            modmsg_debug( "  modifier particle space: <" << (fModifierParticle->GetCurrentSpace() ? fModifierParticle->GetCurrentSpace()->GetName() : "" ) << ">" << eom )
            modmsg_debug( "  modifier particle surface: <" << (fModifierParticle->GetCurrentSurface() ? fModifierParticle->GetCurrentSurface()->GetName() : "" ) << ">" << eom )
            modmsg_debug( "  modifier particle time: <" << fModifierParticle->GetTime() << ">" << eom )
            modmsg_debug( "  modifier particle length: <" << fModifierParticle->GetLength() << ">" << eom )
            modmsg_debug( "  modifier particle position: <" << fModifierParticle->GetPosition().X() << ", " << fModifierParticle->GetPosition().Y() << ", " << fModifierParticle->GetPosition().Z() << ">" << eom )
            modmsg_debug( "  modifier particle momentum: <" << fModifierParticle->GetMomentum().X() << ", " << fModifierParticle->GetMomentum().Y() << ", " << fModifierParticle->GetMomentum().Z() << ">" << eom )
            modmsg_debug( "  modifier particle kinetic energy: <" << fModifierParticle->GetKineticEnergy_eV() << ">" << eom )
            modmsg_debug( "  modifier particle electric field: <" << fModifierParticle->GetElectricField().X() << "," << fModifierParticle->GetElectricField().Y() << "," << fModifierParticle->GetElectricField().Z() << ">" << eom )
            modmsg_debug( "  modifier particle magnetic field: <" << fModifierParticle->GetMagneticField().X() << "," << fModifierParticle->GetMagneticField().Y() << "," << fModifierParticle->GetMagneticField().Z() << ">" << eom )
            modmsg_debug( "  modifier particle angle to magnetic field: <" << fModifierParticle->GetPolarAngleToB() << ">" << eom )

            return false; //changes to inital particle state disabled
        }

        fStep->ModifierFlag() = ExecutePreStepModification( *fModifierParticle, *fParticleQueue );

        (void) fStep->ModifierFlag();
        //hasChangedState is unused because we are operating on the modifier particle
        //this disables any changes to the intial particle

        if( fStep->ModifierFlag() == true )
        {
            modmsg_debug( "modifier calculation:" << eom )
            modmsg_debug( "  modification may occur" << eom )
        }
        else
        {
            modmsg_debug( "modifier calculation:" << eom )
            modmsg_debug( "  modification will not occur" << eom )
        }

        modmsg_debug( "modifier calculation modifier particle state: " << eom )
        modmsg_debug( "  modifier particle space: <" << (fModifierParticle->GetCurrentSpace() ? fModifierParticle->GetCurrentSpace()->GetName() : "" ) << ">" << eom )
        modmsg_debug( "  modifier particle surface: <" << (fModifierParticle->GetCurrentSurface() ? fModifierParticle->GetCurrentSurface()->GetName() : "" ) << ">" << eom )
        modmsg_debug( "  modifier particle time: <" << fModifierParticle->GetTime() << ">" << eom )
        modmsg_debug( "  modifier particle length: <" << fModifierParticle->GetLength() << ">" << eom )
        modmsg_debug( "  modifier particle position: <" << fModifierParticle->GetPosition().X() << ", " << fModifierParticle->GetPosition().Y() << ", " << fModifierParticle->GetPosition().Z() << ">" << eom )
        modmsg_debug( "  modifier particle momentum: <" << fModifierParticle->GetMomentum().X() << ", " << fModifierParticle->GetMomentum().Y() << ", " << fModifierParticle->GetMomentum().Z() << ">" << eom )
        modmsg_debug( "  modifier particle kinetic energy: <" << fModifierParticle->GetKineticEnergy_eV() << ">" << eom )
        modmsg_debug( "  modifier particle electric field: <" << fModifierParticle->GetElectricField().X() << "," << fModifierParticle->GetElectricField().Y() << "," << fModifierParticle->GetElectricField().Z() << ">" << eom )
        modmsg_debug( "  modifier particle magnetic field: <" << fModifierParticle->GetMagneticField().X() << "," << fModifierParticle->GetMagneticField().Y() << "," << fModifierParticle->GetMagneticField().Z() << ">" << eom )
        modmsg_debug( "  modifier particle angle to magnetic field: <" << fModifierParticle->GetPolarAngleToB() << ">" << eom )

        return false; //changes to initial particle state disabled
    }

    bool KSRootStepModifier::ExecutePostStepModification()
    {
        bool hasChangedState = ExecutePostStepModification( *fModifierParticle, *fFinalParticle, *fParticleQueue );
        fFinalParticle->ReleaseLabel( fStep->ModifierName() );

        modmsg_debug( "modifier execution:" << eom )
        modmsg_debug( "  terminator name: <" << fStep->TerminatorName() << ">" << eom )
        modmsg_debug( "  step continuous time: <" << fStep->ContinuousTime() << ">" << eom )
        modmsg_debug( "  step continuous length: <" << fStep->ContinuousLength() << ">" << eom )
        modmsg_debug( "  step continuous energy change: <" << fStep->ContinuousEnergyChange() << ">" << eom )
        modmsg_debug( "  step continuous momentum change: <" << fStep->ContinuousMomentumChange() << ">" << eom )
        modmsg_debug( "  step discrete secondaries: <" << fStep->DiscreteSecondaries() << ">" << eom )
        modmsg_debug( "  step discrete energy change: <" << fStep->DiscreteEnergyChange() << ">" << eom )
        modmsg_debug( "  step discrete momentum change: <" << fStep->DiscreteMomentumChange() << ">" << eom )

        modmsg_debug( "modifier final particle state: " << eom )
        modmsg_debug( "  final particle space: <" << (fModifierParticle->GetCurrentSpace() ? fModifierParticle->GetCurrentSpace()->GetName() : "" ) << ">" << eom )
        modmsg_debug( "  final particle surface: <" << (fModifierParticle->GetCurrentSurface() ? fModifierParticle->GetCurrentSurface()->GetName() : "" ) << ">" << eom )
        modmsg_debug( "  final particle time: <" << fModifierParticle->GetTime() << ">" << eom )
        modmsg_debug( "  final particle length: <" << fModifierParticle->GetLength() << ">" << eom )
        modmsg_debug( "  final particle position: <" << fModifierParticle->GetPosition().X() << ", " << fModifierParticle->GetPosition().Y() << ", " << fModifierParticle->GetPosition().Z() << ">" << eom )
        modmsg_debug( "  final particle momentum: <" << fModifierParticle->GetMomentum().X() << ", " << fModifierParticle->GetMomentum().Y() << ", " << fModifierParticle->GetMomentum().Z() << ">" << eom )
        modmsg_debug( "  final particle kinetic energy: <" << fModifierParticle->GetKineticEnergy_eV() << ">" << eom )
        modmsg_debug( "  final particle electric field: <" << fModifierParticle->GetElectricField().X() << "," << fModifierParticle->GetElectricField().Y() << "," << fModifierParticle->GetElectricField().Z() << ">" << eom )
        modmsg_debug( "  final particle magnetic field: <" << fModifierParticle->GetMagneticField().X() << "," << fModifierParticle->GetMagneticField().Y() << "," << fModifierParticle->GetMagneticField().Z() << ">" << eom )
        modmsg_debug( "  final particle angle to magnetic field: <" << fModifierParticle->GetPolarAngleToB() << ">" << eom )

        return hasChangedState;
    }

    bool KSRootStepModifier::ExecutePreStepModification( KSParticle& anInitialParticle,
                                                         KSParticleQueue& aParticleQueue )
    {
        bool hasChangedState = false;
        for( int tIndex = 0; tIndex < fModifiers.End(); tIndex++ )
        {
            bool changed = fModifiers.ElementAt( tIndex )->ExecutePreStepModification( anInitialParticle, aParticleQueue );
            if(changed){hasChangedState = true;};
        }

        return hasChangedState;
    }

    bool KSRootStepModifier::ExecutePostStepModification( KSParticle& anInitialParticle,
                                                         KSParticle& aFinalParticle,
                                                         KSParticleQueue& aParticleQueue )
    {
        bool hasChangedState = false;
        for( int tIndex = 0; tIndex < fModifiers.End(); tIndex++ )
        {
            bool changed = fModifiers.ElementAt( tIndex )->ExecutePostStepModification( anInitialParticle,
                                                                        aFinalParticle,
                                                                        aParticleQueue );
            if(changed){hasChangedState = true;};
        }

        return hasChangedState;
    }


    STATICINT sKSRootModifierDict =
            KSDictionary< KSRootStepModifier >::AddCommand( &KSRootStepModifier::AddModifier,
                                                            &KSRootStepModifier::RemoveModifier,
                                                            "add_modifier",
                                                            "remove_modifier" );

}
