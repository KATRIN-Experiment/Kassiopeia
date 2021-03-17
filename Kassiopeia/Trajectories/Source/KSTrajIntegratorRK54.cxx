#include "KSTrajIntegratorRK54.h"

namespace Kassiopeia
{

KSTrajIntegratorRK54::KSTrajIntegratorRK54() = default;
KSTrajIntegratorRK54::KSTrajIntegratorRK54(const KSTrajIntegratorRK54&) : KSComponent() {}
KSTrajIntegratorRK54* KSTrajIntegratorRK54::Clone() const
{
    return new KSTrajIntegratorRK54(*this);
}
KSTrajIntegratorRK54::~KSTrajIntegratorRK54() = default;

}  // namespace Kassiopeia
