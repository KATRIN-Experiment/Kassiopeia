#include "KSTermMaxStepTime.h"

#include "KSTerminatorsMessage.h"

#include "KSParticle.h"

namespace Kassiopeia
{

    KSTermMaxStepTime::KSTermMaxStepTime() :
            fTime( 0. ),
            fLastClock( 0 )
    {
    }
    KSTermMaxStepTime::KSTermMaxStepTime( const KSTermMaxStepTime& aCopy ) :
            KSComponent(),
            fTime( aCopy.fTime ),
            fLastClock( aCopy.fLastClock )
    {
    }
    KSTermMaxStepTime* KSTermMaxStepTime::Clone() const
    {
        return new KSTermMaxStepTime( *this );
    }
    KSTermMaxStepTime::~KSTermMaxStepTime()
    {
    }

    void KSTermMaxStepTime::CalculateTermination( const KSParticle& /*anInitialParticle*/, bool& aFlag )
    {
        if (fTime < 0.)
            termmsg( eError ) << "negative time defined in MaxStepTime terminator <" << this->GetName() << ">" << eom;

        std::clock_t tClock = std::clock();
        aFlag = (tClock - fLastClock) / (double)CLOCKS_PER_SEC > fTime ;

        fLastClock = tClock;
        return;
    }
    void KSTermMaxStepTime::ExecuteTermination( const KSParticle&, KSParticle& aFinalParticle, KSParticleQueue& ) const
    {
        aFinalParticle.SetActive( false );
        aFinalParticle.SetLabel(  GetName() );
        return;
    }

}
