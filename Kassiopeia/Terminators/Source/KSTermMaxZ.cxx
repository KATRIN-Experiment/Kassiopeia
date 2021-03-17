#include "KSTermMaxZ.h"

namespace Kassiopeia
{

KSTermMaxZ::KSTermMaxZ() : fMaxZ(0.) {}
KSTermMaxZ::KSTermMaxZ(const KSTermMaxZ& aCopy) : KSComponent(aCopy), fMaxZ(aCopy.fMaxZ) {}
KSTermMaxZ* KSTermMaxZ::Clone() const
{
    return new KSTermMaxZ(*this);
}
KSTermMaxZ::~KSTermMaxZ() = default;

void KSTermMaxZ::CalculateTermination(const KSParticle& anInitialParticle, bool& aFlag)
{
    if (anInitialParticle.GetPosition().Z() > fMaxZ) {
        aFlag = true;
        return;
    }
    aFlag = false;
    return;
}
void KSTermMaxZ::ExecuteTermination(const KSParticle&, KSParticle& aFinalParticle, KSParticleQueue&) const
{
    aFinalParticle.SetActive(false);
    aFinalParticle.SetLabel(GetName());
    return;
}

}  // namespace Kassiopeia
