#include "KSTrajTermGyration.h"
#include "KSTrajectoriesMessage.h"

#include "KConst.h"

namespace Kassiopeia
{

    KSTrajTermGyration::KSTrajTermGyration()
    {
    }
    KSTrajTermGyration::KSTrajTermGyration( const KSTrajTermGyration& )
    {
    }
    KSTrajTermGyration* KSTrajTermGyration::Clone() const
    {
        return new KSTrajTermGyration( *this );
    }
    KSTrajTermGyration::~KSTrajTermGyration()
    {
    }

    void KSTrajTermGyration::Differentiate( const KSTrajAdiabaticParticle& aParticle, KSTrajAdiabaticDerivative& aDerivative ) const
    {
        double tPhaseVelocity = 2. * KConst::Pi() * aParticle.GetCyclotronFrequency();

        aDerivative.AddToPhaseVelocity( tPhaseVelocity );

        return;
    }

}
