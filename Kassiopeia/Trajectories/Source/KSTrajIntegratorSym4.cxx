#include "KSTrajIntegratorSym4.h"

namespace Kassiopeia
{

    KSTrajIntegratorSym4::KSTrajIntegratorSym4()
    {
    }
    KSTrajIntegratorSym4::KSTrajIntegratorSym4( const KSTrajIntegratorSym4& ):
        KSComponent()
    {
    }
    KSTrajIntegratorSym4* KSTrajIntegratorSym4::Clone() const
    {
        return new KSTrajIntegratorSym4( *this );
    }
    KSTrajIntegratorSym4::~KSTrajIntegratorSym4()
    {
    }

}
