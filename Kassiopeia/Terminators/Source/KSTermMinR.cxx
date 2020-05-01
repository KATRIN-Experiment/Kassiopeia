#include "KSTermMinR.h"

#include "KSTerminatorsMessage.h"

namespace Kassiopeia
{

KSTermMinR::KSTermMinR() : fMinR(0.) {}
KSTermMinR::KSTermMinR(const KSTermMinR& aCopy) : KSComponent(), fMinR(aCopy.fMinR) {}
KSTermMinR* KSTermMinR::Clone() const
{
    return new KSTermMinR(*this);
}
KSTermMinR::~KSTermMinR() {}

void KSTermMinR::CalculateTermination(const KSParticle& anInitialParticle, bool& aFlag)
{
    if (fMinR < 0.)
        termmsg(eError) << "negative radius defined in MinR terminator <" << this->GetName() << ">" << eom;

    if (anInitialParticle.GetPosition().Perp() < fMinR) {
        aFlag = true;
        return;
    }
    aFlag = false;
    return;
}
void KSTermMinR::ExecuteTermination(const KSParticle&, KSParticle& aFinalParticle, KSParticleQueue&) const
{
    aFinalParticle.SetActive(false);
    aFinalParticle.SetLabel(GetName());
    return;
}

}  // namespace Kassiopeia
