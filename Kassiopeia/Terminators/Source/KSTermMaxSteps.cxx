#include "KSTermMaxSteps.h"

namespace Kassiopeia
{

    KSTermMaxSteps::KSTermMaxSteps() :
            fMaxSteps( 0 ),
            fSteps( 0 )
    {
    }
    KSTermMaxSteps::KSTermMaxSteps( const KSTermMaxSteps& aCopy ) :
            KSComponent(),
            fMaxSteps( aCopy.fMaxSteps ),
            fSteps( aCopy.fMaxSteps )
    {
    }
    KSTermMaxSteps* KSTermMaxSteps::Clone() const
    {
        return new KSTermMaxSteps( *this );
    }
    KSTermMaxSteps::~KSTermMaxSteps()
    {
    }

    void KSTermMaxSteps::ActivateComponent()
    {
        fSteps = 0;
        return;
    }
    void KSTermMaxSteps::DeactivateComponent()
    {
        fSteps = 0;
        return;
    }

    void KSTermMaxSteps::CalculateTermination( const KSParticle&, bool& aFlag )
    {
        if( fSteps >= fMaxSteps )
        {
            aFlag = true;
            return;
        }
        fSteps++;
        aFlag = false;
        return;
    }
    void KSTermMaxSteps::ExecuteTermination( const KSParticle&, KSParticle& aFinalParticle, KSParticleQueue& ) const
    {
        aFinalParticle.SetActive( false );
        aFinalParticle.SetLabel(  GetName() );
        return;
    }

}
