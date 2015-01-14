#include "KSTermMaxEnergy.h"

#include "KSTerminatorsMessage.h"

namespace Kassiopeia
{

    KSTermMaxEnergy::KSTermMaxEnergy() :
        fMaxEnergy( 0. )
    {
    }
    KSTermMaxEnergy::KSTermMaxEnergy( const KSTermMaxEnergy& aCopy ) :
        fMaxEnergy( aCopy.fMaxEnergy )
    {
    }
    KSTermMaxEnergy* KSTermMaxEnergy::Clone() const
    {
        return new KSTermMaxEnergy( *this );
    }
    KSTermMaxEnergy::~KSTermMaxEnergy()
    {
    }

    void KSTermMaxEnergy::CalculateTermination( const KSParticle& anInitialParticle, bool& aFlag )
    {
        if (fMaxEnergy < 0.)
            termmsg( eError ) << "negative energy defined in MaxEnergy terminator <" << this->GetName() << ">" << eom;

        if( anInitialParticle.GetKineticEnergy_eV() > fMaxEnergy )
        {
            aFlag = true;
            return;
        }
        aFlag = false;
        return;
    }
    void KSTermMaxEnergy::ExecuteTermination( const KSParticle&, KSParticle& aFinalParticle, KSParticleQueue& ) const
    {
        aFinalParticle.SetActive( false );
        aFinalParticle.SetLabel(  GetName() );
        return;
    }

}
