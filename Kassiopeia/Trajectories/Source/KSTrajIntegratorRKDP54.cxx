#include "KSTrajIntegratorRKDP54.h"

namespace Kassiopeia
{

KSTrajIntegratorRKDP54::KSTrajIntegratorRKDP54() = default;
KSTrajIntegratorRKDP54::KSTrajIntegratorRKDP54(const KSTrajIntegratorRKDP54&) : KSComponent() {}
KSTrajIntegratorRKDP54* KSTrajIntegratorRKDP54::Clone() const
{
    return new KSTrajIntegratorRKDP54(*this);
}
KSTrajIntegratorRKDP54::~KSTrajIntegratorRKDP54() = default;

}  // namespace Kassiopeia
