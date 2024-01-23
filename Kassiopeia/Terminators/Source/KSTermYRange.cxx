#include "KSTermYRange.h"

namespace Kassiopeia
{

KSTermYRange::KSTermYRange() : fMinY(0.), fMaxY(0.) {}
KSTermYRange::KSTermYRange(const KSTermYRange& aCopy) : KSComponent(aCopy), fMinY(aCopy.fMinY), fMaxY(aCopy.fMaxY) {}
KSTermYRange* KSTermYRange::Clone() const
{
    return new KSTermYRange(*this);
}
KSTermYRange::~KSTermYRange() = default;

void KSTermYRange::CalculateTermination(const KSParticle& anInitialParticle, bool& aFlag)
{
    if (anInitialParticle.GetPosition().Y() > fMaxY) {
        aFlag = true;
        return;
    }

    if (anInitialParticle.GetPosition().Y() < fMinY) {
        aFlag = true;
        return;
    }

    aFlag = false;
    return;
}
void KSTermYRange::ExecuteTermination(const KSParticle&, KSParticle& aFinalParticle, KSParticleQueue&) const
{
    aFinalParticle.SetActive(false);
    aFinalParticle.SetLabel(GetName());
    return;
}

}  // namespace Kassiopeia
