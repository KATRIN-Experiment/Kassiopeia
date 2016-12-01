#include "KSModDynamicEnhancement.h"
#include "KSModifiersMessage.h"
#include "KSParticleFactory.h"

namespace Kassiopeia
{
    KSModDynamicEnhancement::KSModDynamicEnhancement():
            fEnhancement( 1. ),
            fStaticEnhancement( 1. ),
            fDynamic( false ),
            fReferenceCrossSectionEnergy( -1. ),
            fScattering( NULL ),
            fSynchrotron( NULL ),
            fReferenceCrossSection( 0. )
    {
    }

    KSModDynamicEnhancement::KSModDynamicEnhancement(const KSModDynamicEnhancement &aCopy ):
            KSComponent(),
            fEnhancement( 1. ),
            fStaticEnhancement( aCopy.fStaticEnhancement ),
            fDynamic( aCopy.fDynamic ),
            fReferenceCrossSectionEnergy( aCopy.fReferenceCrossSectionEnergy ),
            fScattering( aCopy.fScattering ),
            fSynchrotron( aCopy.fSynchrotron ),
            fReferenceCrossSection( aCopy.fReferenceCrossSection )
    {
    }

    KSModDynamicEnhancement* KSModDynamicEnhancement::Clone() const
    {
        return new KSModDynamicEnhancement( *this );
    }

    KSModDynamicEnhancement::~KSModDynamicEnhancement()
    {
    }

    bool KSModDynamicEnhancement::ExecutePreStepModification(KSParticle& anInitialParticle, KSParticleQueue& /*aQueue*/)
    {
        double tDynamicEnhancement = 1.;
        if( fDynamic )
        {
            if(anInitialParticle.GetTime() < 1.e-5)
            {
                tDynamicEnhancement = 1.;
            } else {
                double tCrossSection;
                fScattering->CalculateAverageCrossSection(anInitialParticle, anInitialParticle, tCrossSection);
                tDynamicEnhancement = tCrossSection > 0. ? fReferenceCrossSection / tCrossSection : 1.;
            }
        }
        fEnhancement = fStaticEnhancement * tDynamicEnhancement;

        if( fScattering != NULL )
            fScattering->SetEnhancement( fEnhancement );
        if( fSynchrotron != NULL )
            fSynchrotron->SetEnhancement( fEnhancement );

        return false; //intial particle state not changed
    }

    bool KSModDynamicEnhancement::ExecutePostStepModification(KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& /*aQueue*/)
    {
        double tInitialTime = anInitialParticle.GetTime();
        double tFinalTime = aFinalParticle.GetTime();
        double tDuration = tFinalTime - tInitialTime;
        double tEnhancedTime = tInitialTime + tDuration*fStaticEnhancement*fStaticEnhancement;
        aFinalParticle.SetTime(tEnhancedTime);
        return true; //final particle state has changed
    }

    void KSModDynamicEnhancement::SetScattering(KSIntScattering *aScattering)
    {
        fScattering = aScattering;
    }
    void KSModDynamicEnhancement::SetSynchrotron(KSTrajTermSynchrotron *aSynchrotron)
    {
        fSynchrotron = aSynchrotron;
    }

    void KSModDynamicEnhancement::InitializeComponent()
    {
        if( fScattering != NULL && fReferenceCrossSectionEnergy != -1. && fDynamic )
        {
            KSParticle* tInitialisationParticle = KSParticleFactory::GetInstance().Create( 11 );
            tInitialisationParticle->SetPosition(0., 0., 0.);
            tInitialisationParticle->SetMomentum(0, 0, 1.);
            tInitialisationParticle->SetKineticEnergy_eV(fReferenceCrossSectionEnergy);
            tInitialisationParticle->SetTime(0.);

            fScattering->CalculateAverageCrossSection(*tInitialisationParticle, *tInitialisationParticle, fReferenceCrossSection);
            modmsg( eNormal ) << "Initialisation: Dynamic Enhancement Reference CrossSection is: " << fReferenceCrossSection << " at " << fReferenceCrossSectionEnergy << " eV" << eom;

            delete tInitialisationParticle;
        }
    }
    void KSModDynamicEnhancement::DeinitializeComponent()
    {
    }

    void KSModDynamicEnhancement::PullDeupdateComponent()
    {
    }
    void KSModDynamicEnhancement::PushDeupdateComponent()
    {
    }

    STATICINT sKSModDynamicEnhancementDict =
            KSDictionary< KSModDynamicEnhancement >::AddComponent( &KSModDynamicEnhancement::GetEnhancement, "enhancement_factor" );
}
