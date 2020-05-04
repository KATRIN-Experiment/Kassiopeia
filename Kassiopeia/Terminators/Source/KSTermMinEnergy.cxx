#include "KSTermMinEnergy.h"

#include "KSTerminatorsMessage.h"

namespace Kassiopeia
{

KSTermMinEnergy::KSTermMinEnergy() : fMinEnergy(0.) {}
KSTermMinEnergy::KSTermMinEnergy(const KSTermMinEnergy& aCopy) : KSComponent(), fMinEnergy(aCopy.fMinEnergy) {}
KSTermMinEnergy* KSTermMinEnergy::Clone() const
{
    return new KSTermMinEnergy(*this);
}
KSTermMinEnergy::~KSTermMinEnergy() {}

void KSTermMinEnergy::CalculateTermination(const KSParticle& anInitialParticle, bool& aFlag)
{
    if (fMinEnergy < 0.)
        termmsg(eError) << "negative energy defined in MinEnergy terminator <" << this->GetName() << ">" << eom;

    if (anInitialParticle.GetKineticEnergy_eV() < fMinEnergy) {
        aFlag = true;
        return;
    }
    aFlag = false;
    return;
}
void KSTermMinEnergy::ExecuteTermination(const KSParticle&, KSParticle& aFinalParticle, KSParticleQueue&) const
{
    aFinalParticle.SetActive(false);
    aFinalParticle.SetLabel(GetName());
    return;
}

}  // namespace Kassiopeia
