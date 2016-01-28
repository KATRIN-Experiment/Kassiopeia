#include "KSTermMaxLength.h"

#include "KSTerminatorsMessage.h"

#include "KSParticle.h"

namespace Kassiopeia
{

    KSTermMaxLength::KSTermMaxLength() :
            fLength( 0. )
    {
    }
    KSTermMaxLength::KSTermMaxLength( const KSTermMaxLength& aCopy ) :
            KSComponent(),
            fLength( aCopy.fLength )
    {
    }
    KSTermMaxLength* KSTermMaxLength::Clone() const
    {
        return new KSTermMaxLength( *this );
    }
    KSTermMaxLength::~KSTermMaxLength()
    {
    }

    void KSTermMaxLength::CalculateTermination( const KSParticle& anInitialParticle, bool& aFlag )
    {
        if (fLength < 0.)
            termmsg( eError ) << "negative length defined in MaxLength terminator <" << this->GetName() << ">" << eom;

        if( anInitialParticle.GetLength() > fLength )
        {
            aFlag = true;
            return;
        }
        aFlag = false;
        return;
    }
    void KSTermMaxLength::ExecuteTermination( const KSParticle&, KSParticle& aFinalParticle, KSParticleQueue& ) const
    {
        aFinalParticle.SetActive( false );
        aFinalParticle.SetLabel(  GetName() );
        return;
    }

}
