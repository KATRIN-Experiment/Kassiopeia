#include "KSGenSpinRelativeComposite.h"

#include "KSGeneratorsMessage.h"

#include <cmath>

using namespace std;
using KGeoBag::KThreeVector;

namespace Kassiopeia
{

KSGenSpinRelativeComposite::KSGenSpinRelativeComposite() : fThetaValue(nullptr), fPhiValue(nullptr) {}
KSGenSpinRelativeComposite::KSGenSpinRelativeComposite(const KSGenSpinRelativeComposite& aCopy) :
    KSComponent(aCopy),
    fThetaValue(aCopy.fThetaValue),
    fPhiValue(aCopy.fPhiValue)
{}
KSGenSpinRelativeComposite* KSGenSpinRelativeComposite::Clone() const
{
    return new KSGenSpinRelativeComposite(*this);
}
KSGenSpinRelativeComposite::~KSGenSpinRelativeComposite() = default;

void KSGenSpinRelativeComposite::Dice(KSParticleQueue* aPrimaries)
{
    if (!fThetaValue || !fPhiValue)
        genmsg(eError) << "theta or phi value undefined in composite direction creator <" << this->GetName() << ">"
                       << eom;

    KThreeVector tSpin;

    KSParticle* tParticle;
    KSParticleIt tParticleIt;
    KSParticleQueue tParticles;

    double tThetaValue;
    vector<double> tThetaValues;
    vector<double>::iterator tThetaValueIt;

    double tPhiValue;
    std::vector<double> tPhiValues;
    std::vector<double>::iterator tPhiValueIt;

    fThetaValue->DiceValue(tThetaValues);
    fPhiValue->DiceValue(tPhiValues);

    for (tThetaValueIt = tThetaValues.begin(); tThetaValueIt != tThetaValues.end(); tThetaValueIt++) {
        tThetaValue = (katrin::KConst::Pi() / 180.) * (*tThetaValueIt);
        for (tPhiValueIt = tPhiValues.begin(); tPhiValueIt != tPhiValues.end(); tPhiValueIt++) {
            tPhiValue = (katrin::KConst::Pi() / 180.) * (*tPhiValueIt);
            for (tParticleIt = aPrimaries->begin(); tParticleIt != aPrimaries->end(); tParticleIt++) {
                tParticle = new KSParticle(**tParticleIt);
                tParticle->SetAlignedSpin(cos(tThetaValue));
                if (std::isnan(tParticle->GetAlignedSpin())) {
                    tParticle->SetAlignedSpin(1.);
                }
                if (tParticle->GetAlignedSpin() < 0.99999 && tParticle->GetAlignedSpin() > -0.99999) {
                    tParticle->SetSpinAngle(tPhiValue);
                }
                else {
                    tParticle->SetSpinAngle(0);
                }
                tParticle->RecalculateSpinGlobal();
                tParticles.push_back(tParticle);
            }
        }
    }

    for (tParticleIt = aPrimaries->begin(); tParticleIt != aPrimaries->end(); tParticleIt++) {
        tParticle = *tParticleIt;
        delete tParticle;
    }

    aPrimaries->assign(tParticles.begin(), tParticles.end());

    return;
}

void KSGenSpinRelativeComposite::SetThetaValue(KSGenValue* anThetaValue)
{
    if (fThetaValue == nullptr) {
        fThetaValue = anThetaValue;
        return;
    }
    genmsg(eError) << "cannot set theta value <" << anThetaValue->GetName() << "> to composite spin creator <"
                   << this->GetName() << ">" << eom;
    return;
}
void KSGenSpinRelativeComposite::ClearThetaValue(KSGenValue* anThetaValue)
{
    if (fThetaValue == anThetaValue) {
        fThetaValue = nullptr;
        return;
    }
    genmsg(eError) << "cannot clear theta value <" << anThetaValue->GetName() << "> from composite spin creator <"
                   << this->GetName() << ">" << eom;
    return;
}

void KSGenSpinRelativeComposite::SetPhiValue(KSGenValue* aPhiValue)
{
    if (fPhiValue == nullptr) {
        fPhiValue = aPhiValue;
        return;
    }
    genmsg(eError) << "cannot set phi value <" << aPhiValue->GetName() << "> to composite spin creator <"
                   << this->GetName() << ">" << eom;
    return;
}
void KSGenSpinRelativeComposite::ClearPhiValue(KSGenValue* anPhiValue)
{
    if (fPhiValue == anPhiValue) {
        fPhiValue = nullptr;
        return;
    }
    genmsg(eError) << "cannot clear phi value <" << anPhiValue->GetName() << "> from composite spin creator <"
                   << this->GetName() << ">" << eom;
    return;
}

void KSGenSpinRelativeComposite::InitializeComponent()
{
    if (fThetaValue != nullptr) {
        fThetaValue->Initialize();
    }
    if (fPhiValue != nullptr) {
        fPhiValue->Initialize();
    }
    return;
}
void KSGenSpinRelativeComposite::DeinitializeComponent()
{
    if (fThetaValue != nullptr) {
        fThetaValue->Deinitialize();
    }
    if (fPhiValue != nullptr) {
        fPhiValue->Deinitialize();
    }
    return;
}

STATICINT sKSGenDirectionSphericalCompositeDict =
    KSDictionary<KSGenSpinRelativeComposite>::AddCommand(&KSGenSpinRelativeComposite::SetThetaValue,
                                                         &KSGenSpinRelativeComposite::ClearThetaValue, "set_theta",
                                                         "clear_theta") +
    KSDictionary<KSGenSpinRelativeComposite>::AddCommand(
        &KSGenSpinRelativeComposite::SetPhiValue, &KSGenSpinRelativeComposite::ClearPhiValue, "set_phi", "clear_phi");

}  // namespace Kassiopeia
