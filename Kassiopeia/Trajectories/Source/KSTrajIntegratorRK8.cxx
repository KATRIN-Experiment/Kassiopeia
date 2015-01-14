#include "KSTrajIntegratorRK8.h"

namespace Kassiopeia
{

    KSTrajIntegratorRK8::KSTrajIntegratorRK8()
    {
    }
    KSTrajIntegratorRK8::KSTrajIntegratorRK8( const KSTrajIntegratorRK8& )
    {
    }
    KSTrajIntegratorRK8* KSTrajIntegratorRK8::Clone() const
    {
        return new KSTrajIntegratorRK8( *this );
    }
    KSTrajIntegratorRK8::~KSTrajIntegratorRK8()
    {
    }

}
