#include "KSTermTrapped.h"

#include "KSTerminatorsMessage.h"

namespace Kassiopeia
{

	KSTermTrapped::KSTermTrapped() :
            fMaxTurns( 1 ),
            fCurrentTurns( 0 ),
            fCurrentDotProduct( 0 )
    {
    }
	KSTermTrapped::KSTermTrapped( const KSTermTrapped& aCopy ) :
            KSComponent(),
			fMaxTurns( aCopy.fMaxTurns ),
			fCurrentTurns( aCopy.fCurrentTurns ),
			fCurrentDotProduct( aCopy.fCurrentDotProduct )
    {
    }
	KSTermTrapped* KSTermTrapped::Clone() const
    {
        return new KSTermTrapped( *this );
    }
	KSTermTrapped::~KSTermTrapped()
    {
    }

    void KSTermTrapped::ActivateComponent()
    {
    	fCurrentTurns = 0;
    	fCurrentDotProduct = 0;
        return;
    }
    void KSTermTrapped::DeactivateComponent()
    {
    	fCurrentTurns = 0;
    	fCurrentDotProduct = 0;
        return;
    }

    void KSTermTrapped::CalculateTermination( const KSParticle& anInitialParticle, bool& aFlag )
    {
        double DotProduct = anInitialParticle.GetMagneticField().Dot( anInitialParticle.GetMomentum() );

        if( DotProduct * fCurrentDotProduct < 0. )
        {
            fCurrentTurns += 1;
            fCurrentDotProduct = DotProduct;
            if( fCurrentTurns >= fMaxTurns )
            {
                fCurrentTurns = 0;
                fCurrentDotProduct = 0;

                aFlag = true;
                return;
            }
        }

        fCurrentDotProduct = DotProduct;
        aFlag = false;
        return;
    }
    void KSTermTrapped::ExecuteTermination( const KSParticle&, KSParticle& aFinalParticle, KSParticleQueue& ) const
    {
        aFinalParticle.SetActive( false );
        aFinalParticle.SetLabel(  GetName() );
        return;
    }

}
