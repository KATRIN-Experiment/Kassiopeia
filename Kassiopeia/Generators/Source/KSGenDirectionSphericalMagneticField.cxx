#include "KSGenDirectionSphericalMagneticField.h"

#include "KSGeneratorsMessage.h"

using namespace std;

using katrin::KThreeVector;
using katrin::KThreeMatrix;

namespace Kassiopeia
{

KSGenDirectionSphericalMagneticField::KSGenDirectionSphericalMagneticField() :
    fMagneticFields(),
    fThetaValue(nullptr),
    fPhiValue(nullptr)
{}
KSGenDirectionSphericalMagneticField::KSGenDirectionSphericalMagneticField(const KSGenDirectionSphericalMagneticField& aCopy) :
    KSComponent(aCopy),
    fMagneticFields(aCopy.fMagneticFields),
    fThetaValue(aCopy.fThetaValue),
    fPhiValue(aCopy.fPhiValue)
{}
KSGenDirectionSphericalMagneticField* KSGenDirectionSphericalMagneticField::Clone() const
{
    return new KSGenDirectionSphericalMagneticField(*this);
}
KSGenDirectionSphericalMagneticField::~KSGenDirectionSphericalMagneticField() = default;

void KSGenDirectionSphericalMagneticField::Dice(KSParticleQueue* aPrimaries)
{
    if (!fThetaValue || !fPhiValue)
        genmsg(eError) << "theta or phi value undefined in magnetic field direction creator <" << this->GetName() << ">"
                       << eom;

    KThreeVector tMomentum;
    KThreeVector tPosition;
    KThreeVector tMagneticField;

    KThreeVector tXAxis, tYAxis, tZAxis;

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

                tPosition = tParticle->GetPosition();
                CalculateField(tPosition, 0, tMagneticField);

                BuildCoordinateSystem(tXAxis, tYAxis, tZAxis, tMagneticField);

                genmsg(eDebug) << "setting frame to [" << tXAxis << "," << tYAxis << "," << tZAxis << "] from magnetic field " << tMagneticField << " at position " << tPosition << eom;

                tMomentum = tParticle->GetMomentum().Magnitude() *
                            (sin(tThetaValue) * cos(tPhiValue) * tXAxis +
                             sin(tThetaValue) * sin(tPhiValue) * tYAxis +
                             cos(tThetaValue) * tZAxis);
                tParticle->SetMomentum(tMomentum);
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

void KSGenDirectionSphericalMagneticField::BuildCoordinateSystem(KThreeVector& u, KThreeVector& v, KThreeVector& w, const KThreeVector& n)
{
    // this code was taken from https://math.stackexchange.com/questions/542801/rotate-3d-coordinate-system-such-that-z-axis-is-parallel-to-a-given-vector/543538#543538
    static const KThreeVector i_(1, 0, 0);
    static const KThreeVector j_(0, 1, 0);
    static const KThreeVector k_(0, 0, 1);

    const KThreeVector n_ = n.Unit();
    const double theta = acos(k_.Dot(n_));
    const KThreeVector b = k_.Cross(n_);
    const KThreeVector b_ = b.Unit();

    double q0 = cos(theta / 2.);
    double q1 = sin(theta / 2.) * b_.X();
    double q2 = sin(theta / 2.) * b_.Y();
    double q3 = sin(theta / 2.) * b_.Z();

    KThreeMatrix Q(q0*q0 + q1*q1 - q2*q2 - q3*q3,
                   2*(q1*q2 - q0*q3),
                   2*(q1*q3 + q0*q2),
                   2*(q2*q1 + q0*q3),
                   q0*q0 - q1*q1 + q2*q2 - q3*q3,
                   2*(q2*q3 - q0*q1),
                   2*(q3*q1 - q0*q2),
                   2*(q3*q2 + q0*q1),
                   q0*q0 - q1*q1 - q2*q2 + q3*q3);

    u = Q * i_;
    v = Q * j_;
    w = Q * k_;
}

void KSGenDirectionSphericalMagneticField::SetThetaValue(KSGenValue* anThetaValue)
{
    if (fThetaValue == nullptr) {
        fThetaValue = anThetaValue;
        return;
    }
    genmsg(eError) << "cannot set theta value <" << anThetaValue->GetName() << "> to magnetic field direction creator <"
                   << this->GetName() << ">" << eom;
    return;
}
void KSGenDirectionSphericalMagneticField::ClearThetaValue(KSGenValue* anThetaValue)
{
    if (fThetaValue == anThetaValue) {
        fThetaValue = nullptr;
        return;
    }
    genmsg(eError) << "cannot clear theta value <" << anThetaValue->GetName() << "> from magnetic field direction creator <"
                   << this->GetName() << ">" << eom;
    return;
}

void KSGenDirectionSphericalMagneticField::SetPhiValue(KSGenValue* aPhiValue)
{
    if (fPhiValue == nullptr) {
        fPhiValue = aPhiValue;
        return;
    }
    genmsg(eError) << "cannot set phi value <" << aPhiValue->GetName() << "> to magnetic field direction creator <"
                   << this->GetName() << ">" << eom;
    return;
}
void KSGenDirectionSphericalMagneticField::ClearPhiValue(KSGenValue* anPhiValue)
{
    if (fPhiValue == anPhiValue) {
        fPhiValue = nullptr;
        return;
    }
    genmsg(eError) << "cannot clear phi value <" << anPhiValue->GetName() << "> from magnetic field direction creator <"
                   << this->GetName() << ">" << eom;
    return;
}

void KSGenDirectionSphericalMagneticField::AddMagneticField(KSMagneticField* aField)
{
    fMagneticFields.push_back(aField);
}

void KSGenDirectionSphericalMagneticField::CalculateField(const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeVector& aField)
{
    aField = KThreeVector::sZero;
    KThreeVector tCurrentField = KThreeVector::sZero;
    for (auto& magneticField : fMagneticFields) {
        magneticField->CalculateField(aSamplePoint, aSampleTime, tCurrentField);
        aField += tCurrentField;
    }
    return;
}

void KSGenDirectionSphericalMagneticField::InitializeComponent()
{
    if (fThetaValue != nullptr) {
        fThetaValue->Initialize();
    }
    if (fPhiValue != nullptr) {
        fPhiValue->Initialize();
    }
    return;
}
void KSGenDirectionSphericalMagneticField::DeinitializeComponent()
{
    if (fThetaValue != nullptr) {
        fThetaValue->Deinitialize();
    }
    if (fPhiValue != nullptr) {
        fPhiValue->Deinitialize();
    }
    return;
}

STATICINT sKSGenDirectionSphericalMagneticFieldDict =
    KSDictionary<KSGenDirectionSphericalMagneticField>::AddCommand(&KSGenDirectionSphericalMagneticField::SetThetaValue,
                                                               &KSGenDirectionSphericalMagneticField::ClearThetaValue,
                                                               "set_theta", "clear_theta") +
    KSDictionary<KSGenDirectionSphericalMagneticField>::AddCommand(&KSGenDirectionSphericalMagneticField::SetPhiValue,
                                                               &KSGenDirectionSphericalMagneticField::ClearPhiValue,
                                                               "set_phi", "clear_phi");

}  // namespace Kassiopeia
