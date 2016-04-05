#include "KSRootEventModifier.h"
#include "KSModifiersMessage.h"

namespace Kassiopeia
{
    KSRootEventModifier::KSRootEventModifier() :
        fModifiers(128),
        fModifier( NULL ),
        fEvent( NULL )
    {
    }

    KSRootEventModifier::KSRootEventModifier(const KSRootEventModifier &aCopy) : KSComponent(),
        fModifiers( aCopy.fModifiers ),
        fModifier( aCopy.fModifier),
        fEvent( aCopy.fEvent )
    {
    }

    KSRootEventModifier* KSRootEventModifier::Clone() const
    {
        return new KSRootEventModifier( *this );
    }

    KSRootEventModifier::~KSRootEventModifier()
    {
    }

    void KSRootEventModifier::AddModifier(KSEventModifier *aModifier)
    {
        fModifiers.AddElement( aModifier );
        return;
    }
    void KSRootEventModifier::RemoveModifier(KSEventModifier *aModifier)
    {
        fModifiers.RemoveElement( aModifier );
        return;
    }
    void KSRootEventModifier::SetEvent( KSEvent* aEvent )
    {
        fEvent = aEvent;
        return;
    }

    void KSRootEventModifier::PushUpdateComponent()
    {
        for( int tIndex = 0; tIndex < fModifiers.End(); tIndex++ )
        {
            fModifiers.ElementAt( tIndex )->PushUpdate();
        }
    }

    void KSRootEventModifier::PushDeupdateComponent()
    {
        for( int tIndex = 0; tIndex < fModifiers.End(); tIndex++ )
        {
            fModifiers.ElementAt( tIndex )->PushDeupdate();
        }
    }

    bool KSRootEventModifier::ExecutePreEventModification()
    {

        if( fModifiers.End() == 0 )
        {
            modmsg_debug( "modifier calculation:" << eom )
            modmsg_debug( "  no modifier active" << eom )
            //modmsg_debug( "  modifier name: <" << fEvent->GetModifierName() << ">" << eom )
            //modmsg_debug( "  modifier flag: <" << fEvent->GetModificationFlag() << ">" << eom )
/*
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
*/
            return false;
        }

        bool hasChangedState = false;
        for( int tIndex = 0; tIndex < fModifiers.End(); tIndex++ )
        {
            bool changed = fModifiers.ElementAt( tIndex )->ExecutePreEventModification();
            if(changed){hasChangedState = true;};
        }

//        if( fEvent->ModificationFlag() == true )
//        {
//            modmsg_debug( "modifier calculation:" << eom )
//            modmsg_debug( "  modification may occur" << eom )
//        }
//        else
//        {
//            modmsg_debug( "modifier calculation:" << eom )
//            modmsg_debug( "  modification will not occur" << eom )
//        }
/*
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
*/
        return hasChangedState;
    }

    bool KSRootEventModifier::ExecutePostEventModifcation()
    {

        bool hasChangedState = false;
        for( int tIndex = 0; tIndex < fModifiers.End(); tIndex++ )
        {
            bool changed = fModifiers.ElementAt( tIndex )->ExecutePostEventModifcation();
            if(changed){hasChangedState = true;};
        }
/*
        modmsg_debug( "terminator execution:" << eom )
        modmsg_debug( "  terminator name: <" << fEvent->TerminatorName() << ">" << eom )
        modmsg_debug( "  step continuous time: <" << fEvent->ContinuousTime() << ">" << eom )
        modmsg_debug( "  step continuous length: <" << fEvent->ContinuousLength() << ">" << eom )
        modmsg_debug( "  step continuous energy change: <" << fEvent->ContinuousEnergyChange() << ">" << eom )
        modmsg_debug( "  step continuous momentum change: <" << fEvent->ContinuousMomentumChange() << ">" << eom )
        modmsg_debug( "  step discrete secondaries: <" << fEvent->DiscreteSecondaries() << ">" << eom )
        modmsg_debug( "  step discrete energy change: <" << fEvent->DiscreteEnergyChange() << ">" << eom )
        modmsg_debug( "  step discrete momentum change: <" << fEvent->DiscreteMomentumChange() << ">" << eom )

        modmsg_debug( "terminator final particle state: " << eom )
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
*/
        return hasChangedState;
    }

    STATICINT sKSRootModifierDict =
            KSDictionary< KSRootEventModifier >::AddCommand( &KSRootEventModifier::AddModifier,
                                                            &KSRootEventModifier::RemoveModifier,
                                                            "add_modifier",
                                                            "remove_modifier" );

}
