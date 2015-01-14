#include "KSTermMinLongEnergy.h"

#include "KSTerminatorsMessage.h"

namespace Kassiopeia
{

    KSTermMinLongEnergy::KSTermMinLongEnergy() :
            fMinLongEnergy( 0. )
    {
    }
    KSTermMinLongEnergy::KSTermMinLongEnergy( const KSTermMinLongEnergy& aCopy ) :
            fMinLongEnergy( aCopy.fMinLongEnergy )
    {
    }
    KSTermMinLongEnergy* KSTermMinLongEnergy::Clone() const
    {
        return new KSTermMinLongEnergy( *this );
    }
    KSTermMinLongEnergy::~KSTermMinLongEnergy()
    {
    }

    void KSTermMinLongEnergy::CalculateTermination( const KSParticle& anInitialParticle, bool& aFlag )
    {
        if (fMinLongEnergy < 0.)
            termmsg( eError ) << "negative energy defined in MinLongEnergy terminator <" << this->GetName() << ">" << eom;

        if( fabs( anInitialParticle.GetKineticEnergy_eV() * cos( (KConst::Pi() / 180.) * anInitialParticle.GetPolarAngleToB() ) ) < fMinLongEnergy )
        {
            aFlag = true;
            return;
        }
        aFlag = false;
        return;
    }
    void KSTermMinLongEnergy::ExecuteTermination( const KSParticle&, KSParticle& aFinalParticle, KSParticleQueue& ) const
    {
        aFinalParticle.SetActive( false );
        aFinalParticle.SetLabel(  GetName() );
        return;
    }

}
