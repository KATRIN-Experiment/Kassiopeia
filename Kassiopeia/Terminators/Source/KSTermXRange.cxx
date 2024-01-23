#include "KSTermXRange.h"

namespace Kassiopeia
{

KSTermXRange::KSTermXRange() : fMinX(0.), fMaxX(0.) {}
KSTermXRange::KSTermXRange(const KSTermXRange& aCopy) : KSComponent(aCopy), fMinX(aCopy.fMinX), fMaxX(aCopy.fMaxX) {}
KSTermXRange* KSTermXRange::Clone() const
{
    return new KSTermXRange(*this);
}
KSTermXRange::~KSTermXRange() = default;

void KSTermXRange::CalculateTermination(const KSParticle& anInitialParticle, bool& aFlag)
{
    if (anInitialParticle.GetPosition().X() > fMaxX) {
        aFlag = true;
        return;
    }

    if (anInitialParticle.GetPosition().X() < fMinX) {
        aFlag = true;
        return;
    }

    aFlag = false;
    return;
}
void KSTermXRange::ExecuteTermination(const KSParticle&, KSParticle& aFinalParticle, KSParticleQueue&) const
{
    aFinalParticle.SetActive(false);
    aFinalParticle.SetLabel(GetName());
    return;
}

}  // namespace Kassiopeia
