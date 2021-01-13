#include "KSGenLComposite.h"

#include "KSGeneratorsMessage.h"

namespace Kassiopeia
{

KSGenLComposite::KSGenLComposite() : fLValue(nullptr) {}
KSGenLComposite::KSGenLComposite(const KSGenLComposite& aCopy) : KSComponent(aCopy), fLValue(aCopy.fLValue) {}
KSGenLComposite* KSGenLComposite::Clone() const
{
    return new KSGenLComposite(*this);
}
KSGenLComposite::~KSGenLComposite() = default;

void KSGenLComposite::Dice(KSParticleQueue* aPrimaries)
{
    KSParticle* tParticle;
    KSParticleQueue tParticles;
    KSParticleIt tParticleIt;

    double tLValue;
    std::vector<double> tLValues;
    std::vector<double>::iterator tLValueIt;

    fLValue->DiceValue(tLValues);

    for (tLValueIt = tLValues.begin(); tLValueIt != tLValues.end(); tLValueIt++) {
        tLValue = static_cast<int>(*tLValueIt);
        for (tParticleIt = aPrimaries->begin(); tParticleIt != aPrimaries->end(); tParticleIt++) {
            tParticle = new KSParticle(**tParticleIt);
            tParticle->SetSecondQuantumNumber(tLValue);
            tParticles.push_back(tParticle);
        }
    }

    for (tParticleIt = aPrimaries->begin(); tParticleIt != aPrimaries->end(); tParticleIt++) {
        tParticle = *tParticleIt;
        delete tParticle;
    }

    aPrimaries->assign(tParticles.begin(), tParticles.end());

    return;
}

void KSGenLComposite::SetLValue(KSGenValue* anLValue)
{
    if (fLValue == nullptr) {
        fLValue = anLValue;
        return;
    }
    genmsg(eError) << "cannot set L value <" << anLValue->GetName() << "> to composite L creator <" << this->GetName()
                   << ">" << eom;
    return;
}
void KSGenLComposite::ClearLValue(KSGenValue* anLValue)
{
    if (fLValue == anLValue) {
        fLValue = nullptr;
        return;
    }
    genmsg(eError) << "cannot clear L value <" << anLValue->GetName() << "> from composite L creator <"
                   << this->GetName() << ">" << eom;
    return;
}

void KSGenLComposite::InitializeComponent()
{
    if (fLValue != nullptr) {
        fLValue->Initialize();
    }
    return;
}
void KSGenLComposite::DeinitializeComponent()
{
    if (fLValue != nullptr) {
        fLValue->Deinitialize();
    }
    return;
}

STATICINT sKSGenLCompositeDict = KSDictionary<KSGenLComposite>::AddCommand(
    &KSGenLComposite::SetLValue, &KSGenLComposite::ClearLValue, "set_L", "clear_L");

}  // namespace Kassiopeia
