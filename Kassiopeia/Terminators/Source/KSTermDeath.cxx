#include "KSTermDeath.h"

namespace Kassiopeia
{

    KSTermDeath::KSTermDeath()
	{
	}
    KSTermDeath::KSTermDeath( const KSTermDeath& )
    {
    }
    KSTermDeath* KSTermDeath::Clone() const
    {
        return new KSTermDeath( *this );
    }
    KSTermDeath::~KSTermDeath()
	{
	}

    void KSTermDeath::CalculateTermination( const KSParticle& /*anInitialParticle*/, bool& aFlag )
    {
        aFlag = true;
        return;
    }
	void KSTermDeath::ExecuteTermination( const KSParticle&, KSParticle& aFinalParticle, KSParticleQueue& ) const
	{
	    aFinalParticle.SetActive( false );
	    aFinalParticle.SetLabel(  GetName() );
	    return;
	}

}
