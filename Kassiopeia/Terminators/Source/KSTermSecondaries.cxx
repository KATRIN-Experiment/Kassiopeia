#include "KSTermSecondaries.h"

namespace Kassiopeia
{

KSTermSecondaries::KSTermSecondaries() = default;
KSTermSecondaries::KSTermSecondaries(const KSTermSecondaries& /*aCopy*/) : KSComponent() {}
KSTermSecondaries* KSTermSecondaries::Clone() const
{
    return new KSTermSecondaries(*this);
}
KSTermSecondaries::~KSTermSecondaries() = default;

void KSTermSecondaries::CalculateTermination(const KSParticle& anInitialParticle, bool& aFlag)
{
    if (anInitialParticle.GetParentTrackId() != -1) {
        aFlag = true;
        return;
    }
    aFlag = false;
    return;
}

void KSTermSecondaries::ExecuteTermination(const KSParticle&, KSParticle& aFinalParticle, KSParticleQueue&) const
{
    aFinalParticle.SetActive(false);
    aFinalParticle.SetLabel(GetName());
    return;
}


}  // namespace Kassiopeia
