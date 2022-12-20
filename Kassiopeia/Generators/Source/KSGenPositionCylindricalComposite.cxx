#include "KSGenPositionCylindricalComposite.h"

#include "KSGeneratorsMessage.h"

using namespace std;

using katrin::KThreeVector;

namespace Kassiopeia
{

KSGenPositionCylindricalComposite::KSGenPositionCylindricalComposite() :
    fOrigin(KThreeVector::sZero),
    fXAxis(KThreeVector::sXUnit),
    fYAxis(KThreeVector::sYUnit),
    fZAxis(KThreeVector::sZUnit)
{
    fCoordinateMap[eRadius] = 0;
    fCoordinateMap[ePhi] = 1;
    fCoordinateMap[eZ] = 2;
}
KSGenPositionCylindricalComposite::KSGenPositionCylindricalComposite(const KSGenPositionCylindricalComposite& aCopy) :
    KSComponent(aCopy),
    fOrigin(aCopy.fOrigin),
    fXAxis(aCopy.fXAxis),
    fYAxis(aCopy.fYAxis),
    fZAxis(aCopy.fZAxis),
    fCoordinateMap(aCopy.fCoordinateMap),
    fValues(aCopy.fValues)
{}
KSGenPositionCylindricalComposite* KSGenPositionCylindricalComposite::Clone() const
{
    return new KSGenPositionCylindricalComposite(*this);
}
KSGenPositionCylindricalComposite::~KSGenPositionCylindricalComposite() = default;

void KSGenPositionCylindricalComposite::Dice(KSParticleQueue* aPrimaries)
{
    bool tHasRValue = false;
    bool tHasPhiValue = false;
    bool tHasZValue = false;

    for (auto& value : fValues) {
        tHasRValue = tHasRValue | (value.first == eRadius);
        tHasPhiValue = tHasPhiValue | (value.first == ePhi);
        tHasZValue = tHasZValue | (value.first == eZ);
    }

    if (!tHasRValue | !tHasPhiValue | !tHasZValue)
        genmsg(eError) << "r, phi or z value undefined in composite position creator <" << this->GetName() << ">"
                       << eom;

    KThreeVector tPosition;
    KThreeVector tCylindricalPosition;

    KSParticle* tParticle;
    KSParticleIt tParticleIt;
    KSParticleQueue tParticles;

    vector<double> tFirstValues;
    vector<double>::iterator tFirstValueIt;
    vector<double> tSecondValues;
    vector<double>::iterator tSecondValueIt;
    vector<double> tThirdValues;
    vector<double>::iterator tThirdValueIt;

    fValues.at(0).second->DiceValue(tFirstValues);
    fValues.at(1).second->DiceValue(tSecondValues);
    fValues.at(2).second->DiceValue(tThirdValues);

    for (tFirstValueIt = tFirstValues.begin(); tFirstValueIt != tFirstValues.end(); tFirstValueIt++) {
        tCylindricalPosition[fCoordinateMap.at(fValues.at(0).first)] = (*tFirstValueIt);

        for (tSecondValueIt = tSecondValues.begin(); tSecondValueIt != tSecondValues.end(); tSecondValueIt++) {
            tCylindricalPosition[fCoordinateMap.at(fValues.at(1).first)] = (*tSecondValueIt);

            for (tThirdValueIt = tThirdValues.begin(); tThirdValueIt != tThirdValues.end(); tThirdValueIt++) {
                tCylindricalPosition[fCoordinateMap.at(fValues.at(2).first)] = (*tThirdValueIt);

                for (tParticleIt = aPrimaries->begin(); tParticleIt != aPrimaries->end(); tParticleIt++) {
                    double tRValue = tCylindricalPosition[0];
                    double tPhiValue = (katrin::KConst::Pi() / 180.) * tCylindricalPosition[1];
                    double tZValue = tCylindricalPosition[2];

                    tParticle = new KSParticle(**tParticleIt);
                    tPosition = fOrigin;
                    tPosition += tRValue * cos(tPhiValue) * fXAxis;
                    tPosition += tRValue * sin(tPhiValue) * fYAxis;
                    tPosition += tZValue * fZAxis;
                    tParticle->SetPosition(tPosition);
                    tParticles.push_back(tParticle);
                }
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

void KSGenPositionCylindricalComposite::SetOrigin(const KThreeVector& anOrigin)
{
    fOrigin = anOrigin;
    return;
}
void KSGenPositionCylindricalComposite::SetXAxis(const KThreeVector& anXAxis)
{
    fXAxis = anXAxis;
    return;
}
void KSGenPositionCylindricalComposite::SetYAxis(const KThreeVector& anYAxis)
{
    fYAxis = anYAxis;
    return;
}
void KSGenPositionCylindricalComposite::SetZAxis(const KThreeVector& anZAxis)
{
    fZAxis = anZAxis;
    return;
}

void KSGenPositionCylindricalComposite::SetRValue(KSGenValue* anRValue)
{

    for (auto& value : fValues) {
        if (value.first == eRadius) {
            genmsg(eError) << "cannot set r value <" << anRValue->GetName()
                           << "> to composite position cylindrical creator <" << this->GetName() << ">" << eom;
            return;
        }
    }
    fValues.emplace_back(eRadius, anRValue);
}

void KSGenPositionCylindricalComposite::ClearRValue(KSGenValue* anRValue)
{
    for (auto tIt = fValues.begin(); tIt != fValues.end(); tIt++) {
        if ((*tIt).first == eRadius) {
            fValues.erase(tIt);
            return;
        }
    }

    genmsg(eError) << "cannot clear r value <" << anRValue->GetName()
                   << "> from composite position cylindrical creator <" << this->GetName() << ">" << eom;
    return;
}

void KSGenPositionCylindricalComposite::SetPhiValue(KSGenValue* aPhiValue)
{
    for (auto& value : fValues) {
        if (value.first == ePhi) {
            genmsg(eError) << "cannot set phi value <" << aPhiValue->GetName()
                           << "> to composite position cylindrical creator <" << this->GetName() << ">" << eom;
            return;
        }
    }
    fValues.emplace_back(ePhi, aPhiValue);
    return;
}
void KSGenPositionCylindricalComposite::ClearPhiValue(KSGenValue* anPhiValue)
{
    for (auto tIt = fValues.begin(); tIt != fValues.end(); tIt++) {
        if ((*tIt).first == ePhi) {
            fValues.erase(tIt);
            return;
        }
    }

    genmsg(eError) << "cannot clear phi value <" << anPhiValue->GetName()
                   << "> from composite position cylindrical creator <" << this->GetName() << ">" << eom;
    return;
}

void KSGenPositionCylindricalComposite::SetZValue(KSGenValue* anZValue)
{
    for (auto& value : fValues) {
        if (value.first == eZ) {
            genmsg(eError) << "cannot set z value <" << anZValue->GetName()
                           << "> to composite position cylindrical creator <" << this->GetName() << ">" << eom;
            return;
        }
    }
    fValues.emplace_back(eZ, anZValue);
}
void KSGenPositionCylindricalComposite::ClearZValue(KSGenValue* anZValue)
{
    for (auto tIt = fValues.begin(); tIt != fValues.end(); tIt++) {
        if ((*tIt).first == eZ) {
            fValues.erase(tIt);
            return;
        }
    }

    genmsg(eError) << "cannot clear z value <" << anZValue->GetName()
                   << "> from composite position cylindrical creator <" << this->GetName() << ">" << eom;
    return;
}

void KSGenPositionCylindricalComposite::InitializeComponent()
{
    for (auto& value : fValues) {
        value.second->Initialize();
    }
    return;
}
void KSGenPositionCylindricalComposite::DeinitializeComponent()
{
    for (auto& value : fValues) {
        value.second->Deinitialize();
    }
    return;
}

STATICINT sKSGenPositionCylindricalCompositeDict =
    KSDictionary<KSGenPositionCylindricalComposite>::AddCommand(&KSGenPositionCylindricalComposite::SetRValue,
                                                                &KSGenPositionCylindricalComposite::ClearRValue,
                                                                "set_r", "clear_r") +
    KSDictionary<KSGenPositionCylindricalComposite>::AddCommand(&KSGenPositionCylindricalComposite::SetPhiValue,
                                                                &KSGenPositionCylindricalComposite::ClearPhiValue,
                                                                "set_phi", "clear_phi") +
    KSDictionary<KSGenPositionCylindricalComposite>::AddCommand(&KSGenPositionCylindricalComposite::SetZValue,
                                                                &KSGenPositionCylindricalComposite::ClearZValue,
                                                                "set_z", "clear_z");

}  // namespace Kassiopeia
