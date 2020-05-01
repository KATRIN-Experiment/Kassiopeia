#include "KSGenNComposite.h"

#include "KSGeneratorsMessage.h"

namespace Kassiopeia
{

KSGenNComposite::KSGenNComposite() : fNValue(nullptr) {}
KSGenNComposite::KSGenNComposite(const KSGenNComposite& aCopy) : KSComponent(), fNValue(aCopy.fNValue) {}
KSGenNComposite* KSGenNComposite::Clone() const
{
    return new KSGenNComposite(*this);
}
KSGenNComposite::~KSGenNComposite() {}

void KSGenNComposite::Dice(KSParticleQueue* aPrimaries)
{
    KSParticle* tParticle;
    KSParticleQueue tParticles;
    KSParticleIt tParticleIt;

    double tNValue;
    vector<double> tNValues;
    vector<double>::iterator tNValueIt;

    fNValue->DiceValue(tNValues);

    for (tNValueIt = tNValues.begin(); tNValueIt != tNValues.end(); tNValueIt++) {
        tNValue = static_cast<int>(*tNValueIt);
        for (tParticleIt = aPrimaries->begin(); tParticleIt != aPrimaries->end(); tParticleIt++) {
            tParticle = new KSParticle(**tParticleIt);
            tParticle->SetMainQuantumNumber(tNValue);
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

void KSGenNComposite::SetNValue(KSGenValue* anNValue)
{
    if (fNValue == nullptr) {
        fNValue = anNValue;
        return;
    }
    genmsg(eError) << "cannot set n value <" << anNValue->GetName() << "> to composite n creator <" << this->GetName()
                   << ">" << eom;
    return;
}
void KSGenNComposite::ClearNValue(KSGenValue* anNValue)
{
    if (fNValue == anNValue) {
        fNValue = nullptr;
        return;
    }
    genmsg(eError) << "cannot clear n value <" << anNValue->GetName() << "> from composite n creator <"
                   << this->GetName() << ">" << eom;
    return;
}

void KSGenNComposite::InitializeComponent()
{
    if (fNValue != nullptr) {
        fNValue->Initialize();
    }
    return;
}
void KSGenNComposite::DeinitializeComponent()
{
    if (fNValue != nullptr) {
        fNValue->Deinitialize();
    }
    return;
}

STATICINT sKSGenNCompositeDict = KSDictionary<KSGenNComposite>::AddCommand(
    &KSGenNComposite::SetNValue, &KSGenNComposite::ClearNValue, "set_n", "clear_n");

}  // namespace Kassiopeia
