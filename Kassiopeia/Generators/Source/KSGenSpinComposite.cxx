#include "KSGenSpinComposite.h"

#include "KSGeneratorsMessage.h"

#include <cmath>

namespace Kassiopeia
{

KSGenSpinComposite::KSGenSpinComposite() :
    fThetaValue(nullptr),
    fPhiValue(nullptr),
    fXAxis(KThreeVector::sXUnit),
    fYAxis(KThreeVector::sYUnit),
    fZAxis(KThreeVector::sZUnit)
{}
KSGenSpinComposite::KSGenSpinComposite(const KSGenSpinComposite& aCopy) :
    KSComponent(),
    fThetaValue(aCopy.fThetaValue),
    fPhiValue(aCopy.fPhiValue),
    fXAxis(aCopy.fXAxis),
    fYAxis(aCopy.fYAxis),
    fZAxis(aCopy.fZAxis)
{}
KSGenSpinComposite* KSGenSpinComposite::Clone() const
{
    return new KSGenSpinComposite(*this);
}
KSGenSpinComposite::~KSGenSpinComposite() {}

void KSGenSpinComposite::Dice(KSParticleQueue* aPrimaries)
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
    vector<double> tPhiValues;
    vector<double>::iterator tPhiValueIt;

    fThetaValue->DiceValue(tThetaValues);
    fPhiValue->DiceValue(tPhiValues);

    for (tThetaValueIt = tThetaValues.begin(); tThetaValueIt != tThetaValues.end(); tThetaValueIt++) {
        tThetaValue = (katrin::KConst::Pi() / 180.) * (*tThetaValueIt);
        for (tPhiValueIt = tPhiValues.begin(); tPhiValueIt != tPhiValues.end(); tPhiValueIt++) {
            tPhiValue = (katrin::KConst::Pi() / 180.) * (*tPhiValueIt);
            for (tParticleIt = aPrimaries->begin(); tParticleIt != aPrimaries->end(); tParticleIt++) {
                tParticle = new KSParticle(**tParticleIt);
                tSpin = sin(tThetaValue) * cos(tPhiValue) * fXAxis + sin(tThetaValue) * sin(tPhiValue) * fYAxis +
                        cos(tThetaValue) * fZAxis;
                tParticle->SetInitialSpin(tSpin);
                KThreeVector LocalZ = tParticle->GetMagneticField() / tParticle->GetMagneticField().Magnitude();
                KThreeVector LocalX(LocalZ.Z(), 0., -LocalZ.X());
                LocalX = LocalX / LocalX.Magnitude();
                KThreeVector LocalY = LocalZ.Cross(LocalX);

                tParticle->SetAlignedSpin(tSpin.Dot(LocalZ) / tSpin.Magnitude());
                if (std::isnan(tParticle->GetAlignedSpin())) {
                    tParticle->SetAlignedSpin(1.);
                }
                if (tParticle->GetAlignedSpin() < 0.99999 && tParticle->GetAlignedSpin() > -0.99999) {
                    if (tSpin.Dot(LocalY) > 0.) {
                        tParticle->SetSpinAngle(
                            acos(tSpin.Dot(LocalX) / tSpin.Magnitude() /
                                 sqrt(1 - tParticle->GetAlignedSpin() * tParticle->GetAlignedSpin())));
                    }
                    else {
                        tParticle->SetSpinAngle(katrin::KConst::Pi() + acos(tSpin.Dot(LocalX) / tSpin.Magnitude() /
                                                                            sqrt(1 - tParticle->GetAlignedSpin() *
                                                                                         tParticle->GetAlignedSpin())));
                    }
                }
                else {
                    tParticle->SetSpinAngle(0);
                }
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

void KSGenSpinComposite::SetThetaValue(KSGenValue* anThetaValue)
{
    if (fThetaValue == nullptr) {
        fThetaValue = anThetaValue;
        return;
    }
    genmsg(eError) << "cannot set theta value <" << anThetaValue->GetName() << "> to composite spin creator <"
                   << this->GetName() << ">" << eom;
    return;
}
void KSGenSpinComposite::ClearThetaValue(KSGenValue* anThetaValue)
{
    if (fThetaValue == anThetaValue) {
        fThetaValue = nullptr;
        return;
    }
    genmsg(eError) << "cannot clear theta value <" << anThetaValue->GetName() << "> from composite spin creator <"
                   << this->GetName() << ">" << eom;
    return;
}

void KSGenSpinComposite::SetPhiValue(KSGenValue* aPhiValue)
{
    if (fPhiValue == nullptr) {
        fPhiValue = aPhiValue;
        return;
    }
    genmsg(eError) << "cannot set phi value <" << aPhiValue->GetName() << "> to composite spin creator <"
                   << this->GetName() << ">" << eom;
    return;
}
void KSGenSpinComposite::ClearPhiValue(KSGenValue* anPhiValue)
{
    if (fPhiValue == anPhiValue) {
        fPhiValue = nullptr;
        return;
    }
    genmsg(eError) << "cannot clear phi value <" << anPhiValue->GetName() << "> from composite spin creator <"
                   << this->GetName() << ">" << eom;
    return;
}

void KSGenSpinComposite::SetXAxis(const KThreeVector& anXAxis)
{
    fXAxis = anXAxis;
    return;
}
void KSGenSpinComposite::SetYAxis(const KThreeVector& anYAxis)
{
    fYAxis = anYAxis;
    return;
}
void KSGenSpinComposite::SetZAxis(const KThreeVector& anZAxis)
{
    fZAxis = anZAxis;
    return;
}

void KSGenSpinComposite::InitializeComponent()
{
    if (fThetaValue != nullptr) {
        fThetaValue->Initialize();
    }
    if (fPhiValue != nullptr) {
        fPhiValue->Initialize();
    }
    return;
}
void KSGenSpinComposite::DeinitializeComponent()
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
    KSDictionary<KSGenSpinComposite>::AddCommand(&KSGenSpinComposite::SetThetaValue,
                                                 &KSGenSpinComposite::ClearThetaValue, "set_theta", "clear_theta") +
    KSDictionary<KSGenSpinComposite>::AddCommand(&KSGenSpinComposite::SetPhiValue, &KSGenSpinComposite::ClearPhiValue,
                                                 "set_phi", "clear_phi");

}  // namespace Kassiopeia
