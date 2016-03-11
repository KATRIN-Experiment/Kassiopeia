#include "KSTermMinZ.h"

namespace Kassiopeia
{

    KSTermMinZ::KSTermMinZ() :
        fMinZ( 0. )
    {
    }
    KSTermMinZ::KSTermMinZ( const KSTermMinZ& aCopy ) :
        KSComponent(),
        fMinZ( aCopy.fMinZ )
    {
    }
    KSTermMinZ* KSTermMinZ::Clone() const
    {
        return new KSTermMinZ( *this );
    }
    KSTermMinZ::~KSTermMinZ()
    {
    }


    void KSTermMinZ::CalculateTermination( const KSParticle& anInitialParticle, bool& aFlag )
    {
        if( anInitialParticle.GetPosition().Z() < fMinZ )
        {
            aFlag = true;
            return;
        }
        aFlag = false;
        return;
    }
    void KSTermMinZ::ExecuteTermination( const KSParticle&, KSParticle& aFinalParticle, KSParticleQueue& ) const
    {
        aFinalParticle.SetActive( false );
        aFinalParticle.SetLabel(  GetName() );
        return;
    }

}
