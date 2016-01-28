#include "KSTrajIntegratorRK87.h"

namespace Kassiopeia
{

    KSTrajIntegratorRK87::KSTrajIntegratorRK87()
    {
    }
    KSTrajIntegratorRK87::KSTrajIntegratorRK87( const KSTrajIntegratorRK87& ):
        KSComponent()
    {
    }
    KSTrajIntegratorRK87* KSTrajIntegratorRK87::Clone() const
    {
        return new KSTrajIntegratorRK87( *this );
    }
    KSTrajIntegratorRK87::~KSTrajIntegratorRK87()
    {
    }

}
