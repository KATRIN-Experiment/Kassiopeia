#include "KSTrajIntegratorRK86.h"

namespace Kassiopeia
{

    KSTrajIntegratorRK86::KSTrajIntegratorRK86()
    {
    }
    KSTrajIntegratorRK86::KSTrajIntegratorRK86( const KSTrajIntegratorRK86& ):
        KSComponent()
    {
    }
    KSTrajIntegratorRK86* KSTrajIntegratorRK86::Clone() const
    {
        return new KSTrajIntegratorRK86( *this );
    }
    KSTrajIntegratorRK86::~KSTrajIntegratorRK86()
    {
    }

}
