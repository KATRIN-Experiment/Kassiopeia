#include "KSTrajIntegratorRK65.h"

namespace Kassiopeia
{

KSTrajIntegratorRK65::KSTrajIntegratorRK65() {}
KSTrajIntegratorRK65::KSTrajIntegratorRK65(const KSTrajIntegratorRK65&) : KSComponent() {}
KSTrajIntegratorRK65* KSTrajIntegratorRK65::Clone() const
{
    return new KSTrajIntegratorRK65(*this);
}
KSTrajIntegratorRK65::~KSTrajIntegratorRK65() {}

}  // namespace Kassiopeia
