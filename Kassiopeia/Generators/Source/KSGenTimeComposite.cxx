#include "KSGenTimeComposite.h"

#include "KSGeneratorsMessage.h"

namespace Kassiopeia
{

KSGenTimeComposite::KSGenTimeComposite() : fTimeValue(nullptr) {}
KSGenTimeComposite::KSGenTimeComposite(const KSGenTimeComposite& aCopy) :
    KSComponent(aCopy),
    fTimeValue(aCopy.fTimeValue)
{}
KSGenTimeComposite* KSGenTimeComposite::Clone() const
{
    return new KSGenTimeComposite(*this);
}
KSGenTimeComposite::~KSGenTimeComposite() = default;

void KSGenTimeComposite::Dice(KSParticleQueue* aPrimaries)
{
    KSParticle* tParticle;
    KSParticleQueue tParticles;
    KSParticleIt tParticleIt;

    double tTimeValue;
    std::vector<double> tTimeValues;
    std::vector<double>::iterator tTimeValueIt;

    if (!fTimeValue)
        genmsg(eError) << "time value undefined in composite position creator <" << this->GetName() << ">" << eom;

    fTimeValue->DiceValue(tTimeValues);

    for (tTimeValueIt = tTimeValues.begin(); tTimeValueIt != tTimeValues.end(); tTimeValueIt++) {
        tTimeValue = *tTimeValueIt;
        for (tParticleIt = aPrimaries->begin(); tParticleIt != aPrimaries->end(); tParticleIt++) {
            tParticle = new KSParticle(**tParticleIt);
            tParticle->SetTime(tTimeValue);
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

void KSGenTimeComposite::SetTimeValue(KSGenValue* anTimeValue)
{
    if (fTimeValue == nullptr) {
        fTimeValue = anTimeValue;
        return;
    }
    genmsg(eError) << "cannot set time value <" << anTimeValue->GetName() << "> to composite time creator <"
                   << this->GetName() << ">" << eom;
    return;
}
void KSGenTimeComposite::ClearTimeValue(KSGenValue* anTimeValue)
{
    if (fTimeValue == anTimeValue) {
        fTimeValue = nullptr;
        return;
    }
    genmsg(eError) << "cannot clear time value <" << anTimeValue->GetName() << "> from composite time creator <"
                   << this->GetName() << ">" << eom;
    return;
}

void KSGenTimeComposite::InitializeComponent()
{
    if (fTimeValue != nullptr) {
        fTimeValue->Initialize();
    }
    return;
}
void KSGenTimeComposite::DeinitializeComponent()
{
    if (fTimeValue != nullptr) {
        fTimeValue->Deinitialize();
    }
    return;
}

STATICINT sKSGenTimeCompositeDict = KSDictionary<KSGenTimeComposite>::AddCommand(
    &KSGenTimeComposite::SetTimeValue, &KSGenTimeComposite::ClearTimeValue, "set_time", "clear_time");

}  // namespace Kassiopeia
