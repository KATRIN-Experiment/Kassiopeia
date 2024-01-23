#include "KSTermZRange.h"

namespace Kassiopeia
{

KSTermZRange::KSTermZRange() : fMinZ(0.), fMaxZ(0.) {}
KSTermZRange::KSTermZRange(const KSTermZRange& aCopy) : KSComponent(aCopy), fMinZ(aCopy.fMinZ), fMaxZ(aCopy.fMaxZ) {}
KSTermZRange* KSTermZRange::Clone() const
{
    return new KSTermZRange(*this);
}
KSTermZRange::~KSTermZRange() = default;

void KSTermZRange::CalculateTermination(const KSParticle& anInitialParticle, bool& aFlag)
{
    if (anInitialParticle.GetPosition().Z() > fMaxZ) {
        aFlag = true;
        return;
    }

    if (anInitialParticle.GetPosition().Z() < fMinZ) {
        aFlag = true;
        return;
    }

    aFlag = false;
    return;
}
void KSTermZRange::ExecuteTermination(const KSParticle&, KSParticle& aFinalParticle, KSParticleQueue&) const
{
    aFinalParticle.SetActive(false);
    aFinalParticle.SetLabel(GetName());
    return;
}

}  // namespace Kassiopeia
