#include "KSTermMaxTotalTime.h"

#include "KSParticle.h"
#include "KSTerminatorsMessage.h"

namespace Kassiopeia
{

KSTermMaxTotalTime::KSTermMaxTotalTime() : fTime(0.) {}
KSTermMaxTotalTime::KSTermMaxTotalTime(const KSTermMaxTotalTime& aCopy) : KSComponent(aCopy), fTime(aCopy.fTime) {}
KSTermMaxTotalTime* KSTermMaxTotalTime::Clone() const
{
    return new KSTermMaxTotalTime(*this);
}
KSTermMaxTotalTime::~KSTermMaxTotalTime() = default;

void KSTermMaxTotalTime::CalculateTermination(const KSParticle& /*anInitialParticle*/, bool& aFlag)
{
    if (fTime < 0.)
        termmsg(eError) << "negative time defined in MaxStepTime terminator <" << this->GetName() << ">" << eom;

    std::clock_t tClock = std::clock();
    aFlag = tClock / (double) CLOCKS_PER_SEC > fTime;

    return;
}
void KSTermMaxTotalTime::ExecuteTermination(const KSParticle&, KSParticle& aFinalParticle, KSParticleQueue&) const
{
    aFinalParticle.SetActive(false);
    aFinalParticle.SetLabel(GetName());
    return;
}

}  // namespace Kassiopeia
