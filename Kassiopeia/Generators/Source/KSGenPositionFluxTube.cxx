#include "KSGenPositionFluxTube.h"

#include "KConst.h"
#include "KRandom.h"
#include "KSGeneratorsMessage.h"
#include "KSParticle.h"
using katrin::KRandom;

using katrin::KThreeVector;

using namespace std;

namespace Kassiopeia
{

KSGenPositionFluxTube::KSGenPositionFluxTube() :
    fPhiValue(nullptr),
    fZValue(nullptr),
    fMagneticFields(),
    fFlux(0.0191),
    fNIntegrationSteps(1000),
    fOnlySurface(true)
{}
KSGenPositionFluxTube::KSGenPositionFluxTube(const KSGenPositionFluxTube& aCopy) :
    KSComponent(aCopy),
    fPhiValue(aCopy.fPhiValue),
    fZValue(aCopy.fZValue),
    fMagneticFields(aCopy.fMagneticFields),
    fFlux(aCopy.fFlux),
    fNIntegrationSteps(aCopy.fNIntegrationSteps),
    fOnlySurface(aCopy.fOnlySurface)
{}
KSGenPositionFluxTube* KSGenPositionFluxTube::Clone() const
{
    return new KSGenPositionFluxTube(*this);
}
KSGenPositionFluxTube::~KSGenPositionFluxTube() = default;

void KSGenPositionFluxTube::Dice(KSParticleQueue* aPrimaries)
{
    if (!fPhiValue | !fZValue)
        genmsg(eError) << "phi or z value undefined in composite position creator <" << this->GetName() << ">" << eom;

    KThreeVector tPosition;

    KSParticle* tParticle;
    KSParticleIt tParticleIt;
    KSParticleQueue tParticles;

    double tPhiValue;
    vector<double> tPhiValues;
    vector<double>::iterator tPhiValueIt;

    double tZValue;
    vector<double> tZValues;
    vector<double>::iterator tZValueIt;

    double tRValue;
    double tX;
    double tY;

    double tFlux;
    double tArea;
    double tLastArea;

    fPhiValue->DiceValue(tPhiValues);
    fZValue->DiceValue(tZValues);

    for (tZValueIt = tZValues.begin(); tZValueIt != tZValues.end(); tZValueIt++) {
        tZValue = (*tZValueIt);

        for (tPhiValueIt = tPhiValues.begin(); tPhiValueIt != tPhiValues.end(); tPhiValueIt++) {
            tPhiValue = (katrin::KConst::Pi() / 180.) * (*tPhiValueIt);

            tRValue = 0.0;
            tFlux = 0.0;
            tLastArea = 0.0;

            KThreeVector tField;
            //calculate position at z=0 to get approximation for radius
            CalculateField(KThreeVector(0, 0, tZValue), 0.0, tField);
            double tRApproximation = sqrt(fFlux / (katrin::KConst::Pi() * tField.Magnitude()));
            genmsg_debug("r approximation is <" << tRApproximation << ">" << eom);

            //calculate stepsize from 0 to rApproximation
            double tStepSize = tRApproximation / fNIntegrationSteps;

            while (tFlux < fFlux) {
                tX = tRValue * cos(tPhiValue);
                tY = tRValue * sin(tPhiValue);
                CalculateField(KThreeVector(tX, tY, tZValue), 0.0, tField);

                tArea = katrin::KConst::Pi() * tRValue * tRValue;
                tFlux += tField.Magnitude() * (tArea - tLastArea);

                genmsg_debug("r <" << tRValue << ">" << eom);
                genmsg_debug("field " << tField << eom);
                genmsg_debug("area <" << tArea << ">" << eom);
                genmsg_debug("flux <" << tFlux << ">" << eom);

                tRValue += tStepSize;
                tLastArea = tArea;
            }

            //correct the last step, to get a tFlux = fFlux
            tRValue = sqrt(tRValue * tRValue - (tFlux - fFlux) / (tField.Magnitude() * katrin::KConst::Pi()));

            //dice r value if volume option is choosen
            if (!fOnlySurface) {
                tRValue = pow(KRandom::GetInstance().Uniform(0.0, tRValue * tRValue), (1. / 2.));
            }

            genmsg_debug("flux tube generator using r of <" << tRValue << ">" << eom);

            for (tParticleIt = aPrimaries->begin(); tParticleIt != aPrimaries->end(); tParticleIt++) {
                tParticle = new KSParticle(**tParticleIt);
                tPosition = tRValue * cos(tPhiValue) * KThreeVector::sXUnit +
                            tRValue * sin(tPhiValue) * KThreeVector::sYUnit + tZValue * KThreeVector::sZUnit;
                tParticle->SetPosition(tPosition);
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


void KSGenPositionFluxTube::SetPhiValue(KSGenValue* aPhiValue)
{
    if (fPhiValue == nullptr) {
        fPhiValue = aPhiValue;
        return;
    }
    genmsg(eError) << "cannot set phi value <" << aPhiValue->GetName()
                   << "> to composite position cylindrical creator <" << this->GetName() << ">" << eom;
    return;
}
void KSGenPositionFluxTube::ClearPhiValue(KSGenValue* anPhiValue)
{
    if (fPhiValue == anPhiValue) {
        fPhiValue = nullptr;
        return;
    }
    genmsg(eError) << "cannot clear phi value <" << anPhiValue->GetName()
                   << "> from composite position cylindrical creator <" << this->GetName() << ">" << eom;
    return;
}

void KSGenPositionFluxTube::SetZValue(KSGenValue* anZValue)
{
    if (fZValue == nullptr) {
        fZValue = anZValue;
        return;
    }
    genmsg(eError) << "cannot set z value <" << anZValue->GetName() << "> to composite position cylindrical creator <"
                   << this->GetName() << ">" << eom;
    return;
}
void KSGenPositionFluxTube::ClearZValue(KSGenValue* anZValue)
{
    if (fZValue == anZValue) {
        fZValue = nullptr;
        return;
    }
    genmsg(eError) << "cannot clear z value <" << anZValue->GetName()
                   << "> from composite position cylindrical creator <" << this->GetName() << ">" << eom;
    return;
}

void KSGenPositionFluxTube::AddMagneticField(KSMagneticField* aField)
{
    fMagneticFields.push_back(aField);
}

void KSGenPositionFluxTube::CalculateField(const KThreeVector& aSamplePoint, const double& aSampleTime,
                                           KThreeVector& aField)
{
    aField = KThreeVector::sZero;
    KThreeVector tCurrentField = KThreeVector::sZero;
    for (auto& magneticField : fMagneticFields) {
        magneticField->CalculateField(aSamplePoint, aSampleTime, tCurrentField);
        aField += tCurrentField;
    }
    return;
}

void KSGenPositionFluxTube::InitializeComponent()
{
    if (fPhiValue != nullptr) {
        fPhiValue->Initialize();
    }
    if (fZValue != nullptr) {
        fZValue->Initialize();
    }
    for (auto tIndex : fMagneticFields) {
        tIndex->Initialize();
    }
    return;
}
void KSGenPositionFluxTube::DeinitializeComponent()
{
    if (fPhiValue != nullptr) {
        fPhiValue->Deinitialize();
    }
    if (fZValue != nullptr) {
        fZValue->Deinitialize();
    }
    for (auto tIndex : fMagneticFields) {
        tIndex->Deinitialize();
    }
    return;
}

}  // namespace Kassiopeia
