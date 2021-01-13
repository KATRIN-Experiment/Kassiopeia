#include "KSTermMaxLongEnergy.h"

#include "KSTerminatorsMessage.h"

namespace Kassiopeia
{

KSTermMaxLongEnergy::KSTermMaxLongEnergy() : fMaxLongEnergy(0.) {}
KSTermMaxLongEnergy::KSTermMaxLongEnergy(const KSTermMaxLongEnergy& aCopy) :
    KSComponent(aCopy),
    fMaxLongEnergy(aCopy.fMaxLongEnergy)
{}
KSTermMaxLongEnergy* KSTermMaxLongEnergy::Clone() const
{
    return new KSTermMaxLongEnergy(*this);
}
KSTermMaxLongEnergy::~KSTermMaxLongEnergy() = default;

void KSTermMaxLongEnergy::CalculateTermination(const KSParticle& anInitialParticle, bool& aFlag)
{
    if (fMaxLongEnergy < 0.)
        termmsg(eError) << "negative energy defined in MaxLongEnergy terminator <" << this->GetName() << ">" << eom;

    if (fabs(anInitialParticle.GetKineticEnergy_eV() *
             cos((katrin::KConst::Pi() / 180.) * anInitialParticle.GetPolarAngleToB())) > fMaxLongEnergy) {
        aFlag = true;
        return;
    }
    aFlag = false;
    return;
}
void KSTermMaxLongEnergy::ExecuteTermination(const KSParticle&, KSParticle& aFinalParticle, KSParticleQueue&) const
{
    aFinalParticle.SetActive(false);
    aFinalParticle.SetLabel(GetName());
    return;
}

}  // namespace Kassiopeia
