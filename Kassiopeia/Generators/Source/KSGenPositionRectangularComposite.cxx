#include "KSGenPositionRectangularComposite.h"

#include "KSGeneratorsMessage.h"

using namespace std;

namespace Kassiopeia
{

KSGenPositionRectangularComposite::KSGenPositionRectangularComposite() :
    fOrigin(KThreeVector::sZero),
    fXAxis(KThreeVector::sXUnit),
    fYAxis(KThreeVector::sYUnit),
    fZAxis(KThreeVector::sZUnit)
{
    fCoordinateMap[eX] = 0;
    fCoordinateMap[eY] = 1;
    fCoordinateMap[eZ] = 2;
}
KSGenPositionRectangularComposite::KSGenPositionRectangularComposite(const KSGenPositionRectangularComposite& aCopy) :
    KSComponent(),
    fOrigin(aCopy.fOrigin),
    fXAxis(aCopy.fXAxis),
    fYAxis(aCopy.fYAxis),
    fZAxis(aCopy.fZAxis),
    fCoordinateMap(aCopy.fCoordinateMap),
    fValues(aCopy.fValues)
{}
KSGenPositionRectangularComposite* KSGenPositionRectangularComposite::Clone() const
{
    return new KSGenPositionRectangularComposite(*this);
}
KSGenPositionRectangularComposite::~KSGenPositionRectangularComposite() {}

void KSGenPositionRectangularComposite::Dice(KSParticleQueue* aPrimaries)
{
    bool tXValue = false;
    bool tYValue = false;
    bool tZValue = false;

    for (auto tIt = fValues.begin(); tIt != fValues.end(); tIt++) {
        tXValue = tXValue | ((*tIt).first == eX);
        tYValue = tYValue | ((*tIt).first == eY);
        tZValue = tZValue | ((*tIt).first == eZ);
    }

    if (!tXValue | !tYValue | !tZValue)
        genmsg(eError) << "x, y or z value undefined in composite position creator <" << this->GetName() << ">" << eom;

    KThreeVector tPosition;
    KThreeVector tRectangularPosition;

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
        tRectangularPosition[fCoordinateMap.at(fValues.at(0).first)] = (*tFirstValueIt);

        for (tSecondValueIt = tSecondValues.begin(); tSecondValueIt != tSecondValues.end(); tSecondValueIt++) {
            tRectangularPosition[fCoordinateMap.at(fValues.at(1).first)] = (*tSecondValueIt);

            for (tThirdValueIt = tThirdValues.begin(); tThirdValueIt != tThirdValues.end(); tThirdValueIt++) {
                tRectangularPosition[fCoordinateMap.at(fValues.at(2).first)] = (*tThirdValueIt);

                for (tParticleIt = aPrimaries->begin(); tParticleIt != aPrimaries->end(); tParticleIt++) {
                    tParticle = new KSParticle(**tParticleIt);
                    tPosition = fOrigin;
                    tPosition += tRectangularPosition[0] * fXAxis;
                    tPosition += tRectangularPosition[1] * fYAxis;
                    tPosition += tRectangularPosition[2] * fZAxis;
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

void KSGenPositionRectangularComposite::SetXValue(KSGenValue* anXValue)
{
    for (auto tIt = fValues.begin(); tIt != fValues.end(); tIt++) {
        if ((*tIt).first == eX) {
            genmsg(eError) << "cannot set x value <" << anXValue->GetName()
                           << "> to composite position rectangular creator <" << this->GetName() << ">" << eom;
            return;
        }
    }
    fValues.push_back(std::pair<CoordinateType, KSGenValue*>(eX, anXValue));
}
void KSGenPositionRectangularComposite::ClearXValue(KSGenValue* anXValue)
{
    for (auto tIt = fValues.begin(); tIt != fValues.end(); tIt++) {
        if ((*tIt).first == eX) {
            fValues.erase(tIt);
            return;
        }
    }

    genmsg(eError) << "cannot clear x value <" << anXValue->GetName()
                   << "> from composite position rectangular creator <" << this->GetName() << ">" << eom;
    return;
}

void KSGenPositionRectangularComposite::SetYValue(KSGenValue* aYValue)
{
    for (auto tIt = fValues.begin(); tIt != fValues.end(); tIt++) {
        if ((*tIt).first == eY) {
            genmsg(eError) << "cannot set y value <" << aYValue->GetName()
                           << "> to composite position rectangular creator <" << this->GetName() << ">" << eom;
            return;
        }
    }
    fValues.push_back(std::pair<CoordinateType, KSGenValue*>(eY, aYValue));
}
void KSGenPositionRectangularComposite::ClearYValue(KSGenValue* anYValue)
{
    for (auto tIt = fValues.begin(); tIt != fValues.end(); tIt++) {
        if ((*tIt).first == eY) {
            fValues.erase(tIt);
            return;
        }
    }

    genmsg(eError) << "cannot clear y value <" << anYValue->GetName()
                   << "> from composite position rectangular creator <" << this->GetName() << ">" << eom;
    return;
}

void KSGenPositionRectangularComposite::SetZValue(KSGenValue* anZValue)
{
    for (auto tIt = fValues.begin(); tIt != fValues.end(); tIt++) {
        if ((*tIt).first == eZ) {
            genmsg(eError) << "cannot set z value <" << anZValue->GetName()
                           << "> to composite position rectangular creator <" << this->GetName() << ">" << eom;
            return;
        }
    }
    fValues.push_back(std::pair<CoordinateType, KSGenValue*>(eZ, anZValue));
}
void KSGenPositionRectangularComposite::ClearZValue(KSGenValue* anZValue)
{
    for (auto tIt = fValues.begin(); tIt != fValues.end(); tIt++) {
        if ((*tIt).first == eZ) {
            fValues.erase(tIt);
            return;
        }
    }

    genmsg(eError) << "cannot clear z value <" << anZValue->GetName()
                   << "> from composite position rectangular creator <" << this->GetName() << ">" << eom;
    return;
}

void KSGenPositionRectangularComposite::SetOrigin(const KThreeVector& anOrigin)
{
    fOrigin = anOrigin;
    return;
}
void KSGenPositionRectangularComposite::SetXAxis(const KThreeVector& anXAxis)
{
    fXAxis = anXAxis;
    return;
}
void KSGenPositionRectangularComposite::SetYAxis(const KThreeVector& anYAxis)
{
    fYAxis = anYAxis;
    return;
}
void KSGenPositionRectangularComposite::SetZAxis(const KThreeVector& anZAxis)
{
    fZAxis = anZAxis;
    return;
}

void KSGenPositionRectangularComposite::InitializeComponent()
{
    for (auto tIt = fValues.begin(); tIt != fValues.end(); tIt++) {
        (*tIt).second->Initialize();
    }
    return;
}
void KSGenPositionRectangularComposite::DeinitializeComponent()
{
    for (auto tIt = fValues.begin(); tIt != fValues.end(); tIt++) {
        (*tIt).second->Deinitialize();
    }
    return;
}

STATICINT sKSGenPositionRectangularCompositeDict =
    KSDictionary<KSGenPositionRectangularComposite>::AddCommand(&KSGenPositionRectangularComposite::SetXValue,
                                                                &KSGenPositionRectangularComposite::ClearXValue,
                                                                "set_x", "clear_x") +
    KSDictionary<KSGenPositionRectangularComposite>::AddCommand(&KSGenPositionRectangularComposite::SetYValue,
                                                                &KSGenPositionRectangularComposite::ClearYValue,
                                                                "set_y", "clear_y") +
    KSDictionary<KSGenPositionRectangularComposite>::AddCommand(&KSGenPositionRectangularComposite::SetZValue,
                                                                &KSGenPositionRectangularComposite::ClearZValue,
                                                                "set_z", "clear_z");

}  // namespace Kassiopeia
