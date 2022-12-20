#include "KSGenMomentumRectangularComposite.h"

#include "KSGeneratorsMessage.h"

using namespace std;

using katrin::KThreeVector;

namespace Kassiopeia
{

KSGenMomentumRectangularComposite::KSGenMomentumRectangularComposite() :
    fXValue(nullptr),
    fYValue(nullptr),
    fZValue(nullptr),
    fXAxis(KThreeVector::sXUnit),
    fYAxis(KThreeVector::sYUnit),
    fZAxis(KThreeVector::sZUnit)
{}
KSGenMomentumRectangularComposite::KSGenMomentumRectangularComposite(const KSGenMomentumRectangularComposite& aCopy) :
    KSComponent(aCopy),
    fXValue(aCopy.fXValue),
    fYValue(aCopy.fYValue),
    fZValue(aCopy.fZValue),
    fXAxis(aCopy.fXAxis),
    fYAxis(aCopy.fYAxis),
    fZAxis(aCopy.fZAxis)
{}
KSGenMomentumRectangularComposite* KSGenMomentumRectangularComposite::Clone() const
{
    return new KSGenMomentumRectangularComposite(*this);
}
KSGenMomentumRectangularComposite::~KSGenMomentumRectangularComposite() = default;

void KSGenMomentumRectangularComposite::Dice(KSParticleQueue* aPrimaries)
{
    if (!fXValue || !fYValue || !fZValue)
        genmsg(eError) << "x,y or z value undefined in composite direction creator <" << this->GetName() << ">" << eom;

    KSParticle* tParticle;
    KSParticleIt tParticleIt;
    KSParticleQueue tParticles;

    double tXValue;
    vector<double> tXValues;
    vector<double>::iterator tXValueIt;

    double tYValue;
    vector<double> tYValues;
    vector<double>::iterator tYValueIt;

    double tZValue;
    vector<double> tZValues;
    vector<double>::iterator tZValueIt;

    fXValue->DiceValue(tXValues);
    fYValue->DiceValue(tYValues);
    fZValue->DiceValue(tZValues);


    for (tXValueIt = tXValues.begin(); tXValueIt != tXValues.end(); tXValueIt++) {
        tXValue = (*tXValueIt);
        for (tYValueIt = tYValues.begin(); tYValueIt != tYValues.end(); tYValueIt++) {
            tYValue = (*tYValueIt);
            for (tZValueIt = tZValues.begin(); tZValueIt != tZValues.end(); tZValueIt++) {
                tZValue = (*tZValueIt);
                for (tParticleIt = aPrimaries->begin(); tParticleIt != aPrimaries->end(); tParticleIt++) {
                    tParticle = new KSParticle(**tParticleIt);
                    tParticle->SetMomentum(tXValue * fXAxis + tYValue * fYAxis + tZValue * fZAxis);
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

void KSGenMomentumRectangularComposite::SetXValue(KSGenValue* anXValue)
{
    if (fXValue == nullptr) {
        fXValue = anXValue;
        return;
    }
    genmsg(eError) << "cannot set x value <" << anXValue->GetName() << "> to composite direction creator <"
                   << this->GetName() << ">" << eom;
    return;
}
void KSGenMomentumRectangularComposite::ClearXValue(KSGenValue* anXValue)
{
    if (fXValue == anXValue) {
        fXValue = nullptr;
        return;
    }
    genmsg(eError) << "cannot clear x value <" << anXValue->GetName() << "> from composite direction creator <"
                   << this->GetName() << ">" << eom;
    return;
}

void KSGenMomentumRectangularComposite::SetYValue(KSGenValue* aYValue)
{
    if (fYValue == nullptr) {
        fYValue = aYValue;
        return;
    }
    genmsg(eError) << "cannot set y value <" << aYValue->GetName() << "> to composite direction creator <"
                   << this->GetName() << ">" << eom;
    return;
}
void KSGenMomentumRectangularComposite::ClearYValue(KSGenValue* aYValue)
{
    if (fYValue == aYValue) {
        fYValue = nullptr;
        return;
    }
    genmsg(eError) << "cannot clear y value <" << aYValue->GetName() << "> from composite direction creator <"
                   << this->GetName() << ">" << eom;
    return;
}

void KSGenMomentumRectangularComposite::SetZValue(KSGenValue* aZValue)
{
    if (fZValue == nullptr) {
        fZValue = aZValue;
        return;
    }
    genmsg(eError) << "cannot set z value <" << aZValue->GetName() << "> to composite direction creator <"
                   << this->GetName() << ">" << eom;
    return;
}
void KSGenMomentumRectangularComposite::ClearZValue(KSGenValue* aZValue)
{
    if (fYValue == aZValue) {
        fZValue = nullptr;
        return;
    }
    genmsg(eError) << "cannot clear z value <" << aZValue->GetName() << "> from composite direction creator <"
                   << this->GetName() << ">" << eom;
    return;
}


void KSGenMomentumRectangularComposite::SetXAxis(const KThreeVector& anXAxis)
{
    fXAxis = anXAxis;
    return;
}
void KSGenMomentumRectangularComposite::SetYAxis(const KThreeVector& anYAxis)
{
    fYAxis = anYAxis;
    return;
}
void KSGenMomentumRectangularComposite::SetZAxis(const KThreeVector& anZAxis)
{
    fZAxis = anZAxis;
    return;
}

void KSGenMomentumRectangularComposite::InitializeComponent()
{
    if (fXValue != nullptr) {
        fXValue->Initialize();
    }
    if (fYValue != nullptr) {
        fYValue->Initialize();
    }
    if (fZValue != nullptr) {
        fZValue->Initialize();
    }
    return;
}
void KSGenMomentumRectangularComposite::DeinitializeComponent()
{
    if (fXValue != nullptr) {
        fXValue->Deinitialize();
    }
    if (fYValue != nullptr) {
        fYValue->Deinitialize();
    }
    if (fZValue != nullptr) {
        fZValue->Deinitialize();
    }
    return;
}

STATICINT sKSGenMomentumRectangularCompositeDict =
    KSDictionary<KSGenMomentumRectangularComposite>::AddCommand(&KSGenMomentumRectangularComposite::SetXValue,
                                                                &KSGenMomentumRectangularComposite::ClearXValue,
                                                                "set_x", "clear_x") +
    KSDictionary<KSGenMomentumRectangularComposite>::AddCommand(&KSGenMomentumRectangularComposite::SetYValue,
                                                                &KSGenMomentumRectangularComposite::ClearYValue,
                                                                "set_y", "clear_y") +
    KSDictionary<KSGenMomentumRectangularComposite>::AddCommand(&KSGenMomentumRectangularComposite::SetYValue,
                                                                &KSGenMomentumRectangularComposite::ClearYValue,
                                                                "set_z", "clear_z");

}  // namespace Kassiopeia
