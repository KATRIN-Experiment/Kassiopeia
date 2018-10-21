#include "KSModSplitOnTurn.h"
#include "KSModifiersMessage.h"
#include "KSParticleFactory.h"

namespace Kassiopeia
{
    KSModSplitOnTurn::KSModSplitOnTurn():
            fDirection( eForward | eBackward ),
            fCurrentDotProduct( 0. )
    {
    }

    KSModSplitOnTurn::KSModSplitOnTurn(const KSModSplitOnTurn &aCopy ):
            KSComponent(),
            fDirection( aCopy.fDirection ),
            fCurrentDotProduct( aCopy.fCurrentDotProduct )
    {
    }

    KSModSplitOnTurn* KSModSplitOnTurn::Clone() const
    {
        return new KSModSplitOnTurn( *this );
    }

    KSModSplitOnTurn::~KSModSplitOnTurn()
    {
    }

    bool KSModSplitOnTurn::ExecutePreStepModification(KSParticle& /*anInitialParticle*/, KSParticleQueue& /*aQueue*/)
    {
        return false; //intial particle state not changed
    }

    bool KSModSplitOnTurn::ExecutePostStepModification(KSParticle& /*anInitialParticle*/, KSParticle& aFinalParticle, KSParticleQueue& aQueue)
    {
        double DotProduct = aFinalParticle.GetMagneticField().Dot( aFinalParticle.GetMomentum() );

        if( DotProduct * fCurrentDotProduct < 0. )
        {
            if ( ((fDirection & eForward) && fCurrentDotProduct > 0.) || ((fDirection & eBackward) &&  fCurrentDotProduct < 0.) )
            {
                fCurrentDotProduct = DotProduct;

                KSParticle* tSplitParticle = new KSParticle( aFinalParticle );
                tSplitParticle->ResetFieldCaching();
                aQueue.push_back( tSplitParticle );

                aFinalParticle.SetLabel( GetName() );
                aFinalParticle.SetActive( false );

                return true; //final particle state has changed
            }
        }

        fCurrentDotProduct = DotProduct;
        return false; //final particle state has not changed
    }

    void KSModSplitOnTurn::InitializeComponent()
    {
    }
    void KSModSplitOnTurn::DeinitializeComponent()
    {
    }

    void KSModSplitOnTurn::PullDeupdateComponent()
    {
    }
    void KSModSplitOnTurn::PushDeupdateComponent()
    {
    }

    //STATICINT sKSModSplitOnTurnDict =
    //        KSDictionary< KSModSplitOnTurn >::AddComponent( &KSModSplitOnTurn::GetEnhancement, "enhancement_factor" );
}
