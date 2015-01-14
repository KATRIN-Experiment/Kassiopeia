#include "KSNavSurface.h"

#include "KSNavigatorsMessage.h"

namespace Kassiopeia
{

    KSNavSurface::KSNavSurface() :
            fTransmissionSplit( false ),
            fReflectionSplit( false )
    {
    }
    KSNavSurface::KSNavSurface( const KSNavSurface& aCopy ) :
            fTransmissionSplit( aCopy.fTransmissionSplit ),
            fReflectionSplit( aCopy.fReflectionSplit )
    {
    }
    KSNavSurface* KSNavSurface::Clone() const
    {
        return new KSNavSurface( *this );
    }
    KSNavSurface::~KSNavSurface()
    {
    }

    void KSNavSurface::SetTransmissionSplit( const bool& aTransmissionSplit )
    {
        fTransmissionSplit = aTransmissionSplit;
        return;
    }
    const bool& KSNavSurface::GetTransmissionSplit() const
    {
        return fTransmissionSplit;
    }

    void KSNavSurface::SetReflectionSplit( const bool& aReflectionSplit )
    {
        fReflectionSplit = aReflectionSplit;
        return;
    }
    const bool& KSNavSurface::GetReflectionSplit() const
    {
        return fReflectionSplit;
    }

    void KSNavSurface::ExecuteNavigation( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aParticleQueue ) const
    {
        navmsg_debug( "navigation surface <" << this->GetName() << "> executing navigation:" << eom );

        KSSide* tCurrentSide = anInitialParticle.GetCurrentSide();
        KSSurface* tCurrentSurface = anInitialParticle.GetCurrentSurface();
        KSSpace* tCurrentSpace = anInitialParticle.GetCurrentSpace();

        if( tCurrentSurface != NULL )
        {

            navmsg_debug( "  child surface was crossed" << eom );

            aFinalParticle = anInitialParticle;
            aFinalParticle.SetCurrentSide( NULL );
            aFinalParticle.SetCurrentSurface( NULL );
            aFinalParticle.SetCurrentSpace( tCurrentSpace );
            aFinalParticle.SetLabel( GetName() );
            aFinalParticle.AddLabel( tCurrentSurface->GetName() );
            tCurrentSurface->Off();

            if( (fTransmissionSplit == true) || (fReflectionSplit == true) )
            {
                KSParticle* tTransmissionSplitParticle = new KSParticle( aFinalParticle );
                tTransmissionSplitParticle->SetLabel( GetName() );
                tTransmissionSplitParticle->AddLabel( tCurrentSurface->GetName() );
                aParticleQueue.push_back( tTransmissionSplitParticle );
                aFinalParticle.SetActive( false );
            }

            return;
        }

        KThreeVector tMomentum = anInitialParticle.GetMomentum();
        KThreeVector tNormal = anInitialParticle.GetCurrentSide()->Normal( anInitialParticle.GetPosition() );

        if( tCurrentSpace == tCurrentSide->GetInsideParent() )
        {
            if( tMomentum.Dot( tNormal ) > 0. )
            {
                navmsg_debug( "  transmission occurred on boundary <" << tCurrentSide->GetName() << "> of parent space <" << tCurrentSide->GetInsideParent()->GetName() << ">" << eom );

                aFinalParticle = anInitialParticle;
                aFinalParticle.SetCurrentSide( NULL );
                aFinalParticle.SetCurrentSurface( NULL );
                aFinalParticle.SetCurrentSpace( tCurrentSide->GetOutsideParent() );
                aFinalParticle.SetLabel( GetName() );
                aFinalParticle.AddLabel( tCurrentSide->GetName() );
                aFinalParticle.AddLabel( "transmission" );
                aFinalParticle.AddLabel( "outbound" );
                tCurrentSide->Off();
                tCurrentSide->GetInsideParent()->Exit();

                if( fTransmissionSplit == true )
                {
                    KSParticle* tTransmissionSplitParticle = new KSParticle( aFinalParticle );
                    tTransmissionSplitParticle->SetLabel( GetName() );
                    tTransmissionSplitParticle->AddLabel( tCurrentSide->GetName() );
                    tTransmissionSplitParticle->AddLabel( "transmission" );
                    tTransmissionSplitParticle->AddLabel( "outbound" );
                    aParticleQueue.push_back( tTransmissionSplitParticle );
                    aFinalParticle.SetActive( false );
                }

                return;
            }
            else
            {
                navmsg_debug( "  reflection occurred on boundary <" << tCurrentSide->GetName() << "> of parent space <" << tCurrentSide->GetInsideParent()->GetName() << ">" << eom );

                aFinalParticle = anInitialParticle;
                aFinalParticle.SetCurrentSide( NULL );
                aFinalParticle.SetCurrentSurface( NULL );
                aFinalParticle.SetCurrentSpace( tCurrentSide->GetInsideParent() );
                aFinalParticle.SetLabel( GetName() );
                aFinalParticle.AddLabel( tCurrentSide->GetName() );
                aFinalParticle.AddLabel( "reflection" );
                aFinalParticle.AddLabel( "outbound" );
                tCurrentSide->Off();

                if( fReflectionSplit == true )
                {
                    KSParticle* tReflectionSplitParticle = new KSParticle( aFinalParticle );
                    tReflectionSplitParticle->SetLabel( GetName() );
                    tReflectionSplitParticle->AddLabel( tCurrentSide->GetName() );
                    tReflectionSplitParticle->AddLabel( "reflection" );
                    tReflectionSplitParticle->AddLabel( "outbound" );
                    aParticleQueue.push_back( tReflectionSplitParticle );
                    aFinalParticle.SetActive( false );
                }

                return;
            }
        }

        if( tCurrentSpace == tCurrentSide->GetOutsideParent() )
        {
            if( tMomentum.Dot( tNormal ) < 0. )
            {
                navmsg_debug( "  transmission occurred on boundary <" << tCurrentSide->GetName() << "> of child space <" << tCurrentSide->GetInsideParent()->GetName() << ">" << eom );

                aFinalParticle = anInitialParticle;
                aFinalParticle.SetCurrentSide( NULL );
                aFinalParticle.SetCurrentSurface( NULL );
                aFinalParticle.SetCurrentSpace( tCurrentSide->GetInsideParent() );
                aFinalParticle.SetLabel( GetName() );
                aFinalParticle.AddLabel( tCurrentSide->GetName() );
                aFinalParticle.AddLabel( "transmission" );
                aFinalParticle.AddLabel( "inbound" );
                tCurrentSide->Off();
                tCurrentSide->GetInsideParent()->Enter();

                if( fTransmissionSplit == true )
                {
                    KSParticle* tTransmissionSplitParticle = new KSParticle( aFinalParticle );
                    tTransmissionSplitParticle->SetLabel( GetName() );
                    tTransmissionSplitParticle->AddLabel( tCurrentSide->GetName() );
                    tTransmissionSplitParticle->AddLabel( "transmission" );
                    tTransmissionSplitParticle->AddLabel( "inbound" );
                    aParticleQueue.push_back( tTransmissionSplitParticle );
                    aFinalParticle.SetActive( false );
                }

                return;
            }
            else
            {
                navmsg_debug( "  reflection occurred on boundary <" << tCurrentSide->GetName() << "> of child space <" << tCurrentSide->GetInsideParent()->GetName() << ">" << eom );

                aFinalParticle = anInitialParticle;
                aFinalParticle.SetCurrentSide( NULL );
                aFinalParticle.SetCurrentSurface( NULL );
                aFinalParticle.SetCurrentSpace( tCurrentSide->GetOutsideParent() );
                aFinalParticle.SetLabel( GetName() );
                aFinalParticle.AddLabel( tCurrentSide->GetName() );
                aFinalParticle.AddLabel( "reflection" );
                aFinalParticle.AddLabel( "inbound" );
                tCurrentSide->Off();

                if( fReflectionSplit == true )
                {
                    KSParticle* tReflectionSplitParticle = new KSParticle( aFinalParticle );
                    tReflectionSplitParticle->SetLabel( GetName() );
                    tReflectionSplitParticle->AddLabel( tCurrentSide->GetName() );
                    tReflectionSplitParticle->AddLabel( "transmission" );
                    tReflectionSplitParticle->AddLabel( "inbound" );
                    aParticleQueue.push_back( tReflectionSplitParticle );
                    aFinalParticle.SetActive( false );
                }

                return;
            }
        }

        navmsg( eError ) << "could not determine surface navigation" << eom;
        return;
    }
}
