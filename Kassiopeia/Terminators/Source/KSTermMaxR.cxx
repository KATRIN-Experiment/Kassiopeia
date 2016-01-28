#include "KSTermMaxR.h"

#include "KSTerminatorsMessage.h"

namespace Kassiopeia
{

    KSTermMaxR::KSTermMaxR() :
        fMaxR( 0. )
    {
    }
    KSTermMaxR::KSTermMaxR( const KSTermMaxR& aCopy ) :
        KSComponent(),
        fMaxR( aCopy.fMaxR )
    {
    }
    KSTermMaxR* KSTermMaxR::Clone() const
    {
        return new KSTermMaxR( *this );
    }
    KSTermMaxR::~KSTermMaxR()
    {
    }

    void KSTermMaxR::CalculateTermination( const KSParticle& anInitialParticle, bool& aFlag )
    {
        if (fMaxR < 0.)
            termmsg( eError ) << "negative radius defined in MaxR terminator <" << this->GetName() << ">" << eom;

        if( anInitialParticle.GetPosition().Perp() > fMaxR )
        {
            aFlag = true;
            return;
        }
        aFlag = false;
        return;
    }
    void KSTermMaxR::ExecuteTermination( const KSParticle&, KSParticle& aFinalParticle, KSParticleQueue& ) const
    {
        aFinalParticle.SetActive( false );
        aFinalParticle.SetLabel(  GetName() );
        return;
    }

}
