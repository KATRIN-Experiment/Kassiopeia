#include "KSGenPositionHomogeneousFluxTube.h"

#include "KConst.h"
#include "KRandom.h"
#include "KSGeneratorsMessage.h"
#include "KSParticle.h"
using katrin::KRandom;

namespace Kassiopeia
{

KSGenPositionHomogeneousFluxTube::KSGenPositionHomogeneousFluxTube() :
    fMagneticFields(),
    fFlux(0.0191),
    fRmax(0.),
    fNIntegrationSteps(1000),
    fZmin(0.),
    fZmax(0.),
    fPhimin(0.),
    fPhimax(360.)
{}
KSGenPositionHomogeneousFluxTube::KSGenPositionHomogeneousFluxTube(const KSGenPositionHomogeneousFluxTube& aCopy) :
    KSComponent(),
    fMagneticFields(aCopy.fMagneticFields),
    fFlux(aCopy.fFlux),
    fRmax(aCopy.fRmax),
    fNIntegrationSteps(aCopy.fNIntegrationSteps),
    fZmin(aCopy.fZmin),
    fZmax(aCopy.fZmax),
    fPhimin(aCopy.fPhimin),
    fPhimax(aCopy.fPhimax)
{}
KSGenPositionHomogeneousFluxTube* KSGenPositionHomogeneousFluxTube::Clone() const
{
    return new KSGenPositionHomogeneousFluxTube(*this);
}
KSGenPositionHomogeneousFluxTube::~KSGenPositionHomogeneousFluxTube() {}

void KSGenPositionHomogeneousFluxTube::Dice(KSParticleQueue* aPrimaries)
{
    KThreeVector tPosition;

    KSParticle* tParticle;
    KSParticleIt tParticleIt;
    KSParticleQueue tParticles;

    double tPhiValue;

    double tZValue;

    double tRValue;
    double tX;
    double tY;

    double tFlux;
    double tArea;
    double tLastArea;
    double tRref;

    do {

        tZValue = KRandom::GetInstance().Uniform(fZmin, fZmax);
        tPhiValue = KRandom::GetInstance().Uniform(fPhimin * katrin::KConst::Pi() / 180.,
                                                   fPhimax * katrin::KConst::Pi() / 180.);

        tRValue = 0.0;
        tFlux = 0.0;
        tLastArea = 0.0;
        tRref = 0.0;

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

        //dice r value in a cylinder volume with radius tRmax

        tRref = pow(KRandom::GetInstance().Uniform(0.0, fRmax * fRmax), (1. / 2.));

    } while (tRref > tRValue);

    tRValue = tRref;

    genmsg_debug("flux tube generator using r of <" << tRValue << ">" << eom);

    for (tParticleIt = aPrimaries->begin(); tParticleIt != aPrimaries->end(); tParticleIt++) {
        tParticle = new KSParticle(**tParticleIt);
        tPosition = tRValue * cos(tPhiValue) * KThreeVector::sXUnit + tRValue * sin(tPhiValue) * KThreeVector::sYUnit +
                    tZValue * KThreeVector::sZUnit;
        tParticle->SetPosition(tPosition);
        tParticles.push_back(tParticle);
    }


    for (tParticleIt = aPrimaries->begin(); tParticleIt != aPrimaries->end(); tParticleIt++) {
        tParticle = *tParticleIt;
        delete tParticle;
    }

    aPrimaries->assign(tParticles.begin(), tParticles.end());

    return;
}


void KSGenPositionHomogeneousFluxTube::AddMagneticField(KSMagneticField* aField)
{
    fMagneticFields.push_back(aField);
}

void KSGenPositionHomogeneousFluxTube::CalculateField(const KThreeVector& aSamplePoint, const double& aSampleTime,
                                                      KThreeVector& aField)
{
    aField = KThreeVector::sZero;
    KThreeVector tCurrentField = KThreeVector::sZero;
    for (size_t tIndex = 0; tIndex < fMagneticFields.size(); tIndex++) {
        fMagneticFields.at(tIndex)->CalculateField(aSamplePoint, aSampleTime, tCurrentField);
        aField += tCurrentField;
    }
    return;
}

void KSGenPositionHomogeneousFluxTube::InitializeComponent()
{
    if (fRmax < 0) {
        fRmax = fabs(fRmax);
        genmsg(eWarning) << "r_max negative, using absolute value<" << fRmax << ">" << eom;
    }

    if (fZmin == fZmax || fPhimin == fPhimax) {
        genmsg(eWarning) << "z_min and z_max or phi_min and phi_max have the same value" << eom;
    }

    if (fZmin > fZmax) {
        double z_temp = fZmin;
        fZmin = fZmax;
        fZmax = z_temp;

        genmsg(eWarning) << "z_min is greater than z_max, switched values" << eom;
    }

    if (fPhimin > fPhimax) {
        double phi_temp = fPhimin;
        fPhimin = fPhimax;
        fPhimax = phi_temp;

        genmsg(eWarning) << "phi_min is greater than phi_max, switched values" << eom;
    }
    // adding recommended fRmax if left out from user
    if (fRmax == 0.) {
        double tPhiValue;
        double tZValue;

        double tRValue;
        double tRValue_temp;
        double tX;
        double tY;

        double tFlux;
        double tArea;
        double tLastArea;

        tRValue = 0.0;
        tRValue_temp = 0.0;
        tFlux = 0.0;
        tLastArea = 0.0;
        tZValue = 0.0;

        KThreeVector tField;
        CalculateField(KThreeVector(0, 0, tZValue), 0.0, tField);

        for (int i = 0; i <= 100; i++) {

            tZValue = fZmin + (fZmax - fZmin) / (100 - 1) * i;
            for (int j = 0; j <= 36; j++) {
                tPhiValue = fPhimin + (fPhimax - fPhimin) / (36 - 1) * j;
                tPhiValue *= katrin::KConst::Pi() / 180.;

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
                //tRValue = pow( KRandom::GetInstance().Uniform( 0.0, tRValue*tRValue ), (1./2.) );

                if (tRValue_temp < tRValue) {
                    tRValue_temp = tRValue;
                }
            }
        }

        fRmax = 1.1 * tRValue_temp;
        genmsg(eWarning) << "r_max not declared, using recommended value: " << fRmax << eom;
    }

    return;
}
void KSGenPositionHomogeneousFluxTube::DeinitializeComponent()
{
    return;
}

}  // namespace Kassiopeia
