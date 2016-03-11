#include "KSTrajIntegratorRKDP853.h"

namespace Kassiopeia
{

    KSTrajIntegratorRKDP853::KSTrajIntegratorRKDP853()
    {
    }
    KSTrajIntegratorRKDP853::KSTrajIntegratorRKDP853( const KSTrajIntegratorRKDP853& ):
        KSComponent()
    {
    }
    KSTrajIntegratorRKDP853* KSTrajIntegratorRKDP853::Clone() const
    {
        return new KSTrajIntegratorRKDP853( *this );
    }
    KSTrajIntegratorRKDP853::~KSTrajIntegratorRKDP853()
    {
    }

}
