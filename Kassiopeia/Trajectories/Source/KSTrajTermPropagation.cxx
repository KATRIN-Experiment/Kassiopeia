#include "KSTrajTermPropagation.h"

#include "KConst.h"
#include "KSTrajectoriesMessage.h"

#include <cmath>

using KGeoBag::KThreeMatrix;
using KGeoBag::KThreeVector;

namespace Kassiopeia
{

KSTrajTermPropagation::KSTrajTermPropagation() : fDirection(eForward) {}
KSTrajTermPropagation::KSTrajTermPropagation(const KSTrajTermPropagation& aCopy) :
    KSComponent(aCopy),
    fDirection(aCopy.fDirection)
{}
KSTrajTermPropagation* KSTrajTermPropagation::Clone() const
{
    return new KSTrajTermPropagation(*this);
}
KSTrajTermPropagation::~KSTrajTermPropagation() = default;

void KSTrajTermPropagation::Differentiate(double /*aTime*/, const KSTrajExactParticle& aParticle,
                                          KSTrajExactDerivative& aDerivative) const
{
    KThreeVector tVelocity = fDirection * aParticle.GetVelocity();
    KThreeVector tForce =
        aParticle.GetCharge() * (aParticle.GetElectricField() + tVelocity.Cross(aParticle.GetMagneticField()));

    aDerivative.AddToVelocity(tVelocity);
    aDerivative.AddToForce(tForce);

    return;
}
void KSTrajTermPropagation::Differentiate(double /*aTime*/, const KSTrajExactSpinParticle& aParticle,
                                          KSTrajExactSpinDerivative& aDerivative) const
{
    double TempSN = std::sqrt(aParticle.GetSpin().MagnitudeSquared() - aParticle.GetSpin0() * aParticle.GetSpin0());

    KThreeVector tVelocity = fDirection * aParticle.GetVelocity();

    KThreeVector MagneticMoment = aParticle.GetGyromagneticRatio() * aParticle.GetSpin() /
                                  aParticle.GetSpin().Magnitude() * aParticle.GetSpinMagnitude() *
                                  katrin::KConst::Hbar();
    KThreeMatrix MagneticGradient = aParticle.GetMagneticGradient();

    double F1 = -MagneticMoment.X() * MagneticGradient[8] + MagneticMoment.Z() * MagneticGradient[2] -
                MagneticMoment.X() * MagneticGradient[4] + MagneticMoment.Y() * MagneticGradient[1];
    double F2 = -MagneticMoment.Y() * MagneticGradient[0] + MagneticMoment.X() * MagneticGradient[3] -
                MagneticMoment.Y() * MagneticGradient[8] + MagneticMoment.Z() * MagneticGradient[5];
    double F3 = -MagneticMoment.Z() * MagneticGradient[4] + MagneticMoment.Y() * MagneticGradient[7] -
                MagneticMoment.Z() * MagneticGradient[0] + MagneticMoment.X() * MagneticGradient[6];
    KThreeVector DipoleForce(F1, F2, F3);

    KThreeVector tForce =
        aParticle.GetCharge() * (aParticle.GetElectricField() + tVelocity.Cross(aParticle.GetMagneticField())) +
        DipoleForce;

    double SFU;
    {
        // FU
        double Temp0 = aParticle.GetLorentzFactor() * aParticle.GetElectricField().Dot(aParticle.GetVelocity());
        double Temp1 = aParticle.GetLorentzFactor() *
                       (aParticle.GetElectricField().X() * katrin::KConst::C() +
                        aParticle.GetMagneticField().Z() * aParticle.GetVelocity().Y() * katrin::KConst::C() -
                        aParticle.GetMagneticField().Y() * aParticle.GetVelocity().Z() * katrin::KConst::C());
        double Temp2 = aParticle.GetLorentzFactor() *
                       (aParticle.GetElectricField().Y() * katrin::KConst::C() -
                        aParticle.GetMagneticField().Z() * aParticle.GetVelocity().X() * katrin::KConst::C() +
                        aParticle.GetMagneticField().X() * aParticle.GetVelocity().Z() * katrin::KConst::C());
        double Temp3 = aParticle.GetLorentzFactor() *
                       (aParticle.GetElectricField().Z() * katrin::KConst::C() +
                        aParticle.GetMagneticField().Y() * aParticle.GetVelocity().X() * katrin::KConst::C() -
                        aParticle.GetMagneticField().X() * aParticle.GetVelocity().Y() * katrin::KConst::C());

        // SFU
        SFU = Temp0 * aParticle.GetSpin0() - Temp1 * aParticle.GetSpin().X() - Temp2 * aParticle.GetSpin().Y() -
              Temp3 * aParticle.GetSpin().Z();
    }

    // FS
    double FS0 = aParticle.GetElectricField().Dot(aParticle.GetSpin());
    double Temp1 = aParticle.GetElectricField().X() * aParticle.GetSpin0() +
                   aParticle.GetMagneticField().Z() * aParticle.GetSpin().Y() * katrin::KConst::C() -
                   aParticle.GetMagneticField().Y() * aParticle.GetSpin().Z() * katrin::KConst::C();
    double Temp2 = aParticle.GetElectricField().Y() * aParticle.GetSpin0() -
                   aParticle.GetMagneticField().Z() * aParticle.GetSpin().X() * katrin::KConst::C() +
                   aParticle.GetMagneticField().X() * aParticle.GetSpin().Z() * katrin::KConst::C();
    double Temp3 = aParticle.GetElectricField().Z() * aParticle.GetSpin0() +
                   aParticle.GetMagneticField().Y() * aParticle.GetSpin().X() * katrin::KConst::C() -
                   aParticle.GetMagneticField().X() * aParticle.GetSpin().Y() * katrin::KConst::C();
    KThreeVector FS(Temp1, Temp2, Temp3);

    double SDU;
    {
        // dU/dtau
        double DU0 = aParticle.GetLorentzFactor() / aParticle.GetMass() / katrin::KConst::C() *
                     aParticle.GetVelocity().Dot(tForce);
        KThreeVector DU = aParticle.GetLorentzFactor() / aParticle.GetMass() * tForce;
        // SdU/dtau
        SDU = aParticle.GetSpin0() * DU0 - aParticle.GetSpin().Dot(DU);
    }

    double tOmega0 = -aParticle.GetGyromagneticRatio() / katrin::KConst::C() *
                         (FS0 + aParticle.GetLorentzFactor() / katrin::KConst::C() * SFU) -
                     aParticle.GetLorentzFactor() / katrin::KConst::C() * SDU;
    KThreeVector tOmega =
        -aParticle.GetGyromagneticRatio() / katrin::KConst::C() *
            (FS -
             aParticle.GetLorentzFactor() * aParticle.GetVelocity() / katrin::KConst::C() / katrin::KConst::C() * SFU) +
        aParticle.GetLorentzFactor() * aParticle.GetVelocity() / katrin::KConst::C() / katrin::KConst::C() * SDU;

    aDerivative.AddToVelocity(tVelocity);
    aDerivative.AddToForce(tForce / TempSN);
    aDerivative.AddToOmega0(tOmega0 / TempSN);
    aDerivative.AddToOmega(tOmega / TempSN);

    return;
}

void KSTrajTermPropagation::Differentiate(double aTime, const KSTrajExactTrappedParticle& aParticle,
                                          KSTrajExactTrappedDerivative& aDerivative) const
{
    double aTimeStep = aTime;
    if (aTimeStep == 0) {
        KThreeVector tVelocity = aParticle.GetVelocity();
        aDerivative.AddToVelocity(tVelocity);
    }
    else {
        KThreeVector tMagneticField = aParticle.GetMagneticField();
        double tMagneticFieldMagnitude = tMagneticField.Magnitude();
        KThreeVector tMagneticFieldUnit = tMagneticField.Unit();

        KThreeVector tMomentum = aParticle.GetMomentum();
        double aTheta = aParticle.GetCharge() * tMagneticFieldMagnitude * aTimeStep /
                        (aParticle.GetLorentzFactor() * aParticle.GetMass());
        KThreeVector tMomentumMagnetic =
            (-sin(aTheta) * tMagneticFieldUnit.Cross(tMomentum) +
             (1.0 - cos(aTheta)) * (tMagneticFieldUnit.Cross(tMagneticFieldUnit.Cross(tMomentum))));
        KThreeVector tMomentumElectric =
            aParticle.GetCharge() * aParticle.GetElectricField() * aTimeStep * aParticle.GetLorentzFactor();
        if (aTheta != 0) {
            tMomentumElectric +=
                -(1.0 - cos(aTheta)) / aTheta * tMagneticFieldUnit.Cross(tMomentumElectric) +
                (1.0 - sin(aTheta) / aTheta) * tMagneticFieldUnit.Cross(tMagneticFieldUnit.Cross(tMomentumElectric));
        }

        KThreeVector tForce = (tMomentumMagnetic + tMomentumElectric) / aTimeStep;
        aDerivative.AddToForce(tForce);
    }

    return;
}

void KSTrajTermPropagation::Differentiate(double /*aTime*/, const KSTrajAdiabaticParticle& aParticle,
                                          KSTrajAdiabaticDerivative& aDerivative) const
{
    double tLongVelocity = fDirection * aParticle.GetLongVelocity();
    double tLongitudinalMomentum = aParticle.GetLongMomentum();
    double tTransverseMomentum = aParticle.GetTransMomentum();
    double tLorentzFactor = aParticle.GetLorentzFactor();
    double tOrbitalMagneticMoment = aParticle.GetOrbitalMagneticMoment();

    KThreeVector tElectricfield = aParticle.GetElectricField();
    KThreeVector tMagneticField = aParticle.GetMagneticField();
    KThreeMatrix tMagneticGradient = aParticle.GetMagneticGradient();
    KThreeVector tMagneticFieldUnit = tMagneticField.Unit();
    double tMagneticFieldMagnitude = tMagneticField.Magnitude();
    double tMagneticGradientUnit = tMagneticFieldUnit * (tMagneticGradient * tMagneticFieldUnit);

    double tLongitudinalForce = -1. * (tOrbitalMagneticMoment / tLorentzFactor) * tMagneticGradientUnit +
                                aParticle.GetCharge() * tElectricfield.Dot(tMagneticField.Unit());
    double tTransverseForce = ((tLongitudinalMomentum * tTransverseMomentum) /
                               (2 * aParticle.GetMass() * tLorentzFactor * tMagneticFieldMagnitude)) *
                              tMagneticGradientUnit;

    trajmsg_debug("adiabatic propagation gc velocity: <" << tLongVelocity * tMagneticFieldUnit << ">" << ret)
        trajmsg_debug("adiabatic propagation longitudinal force <" << tLongitudinalForce << ">" << ret)
            trajmsg_debug("adiabatic propagation transverse force <" << tTransverseForce << ">" << ret)

                aDerivative.AddToGuidingCenterVelocity(tLongVelocity * tMagneticFieldUnit);
    aDerivative.AddToLongitudinalForce(tLongitudinalForce);
    aDerivative.AddToTransverseForce(tTransverseForce);

    return;
}

void KSTrajTermPropagation::Differentiate(double /*aTime*/, const KSTrajAdiabaticSpinParticle& aParticle,
                                          KSTrajAdiabaticSpinDerivative& aDerivative) const
{

    KThreeVector e3 = aParticle.GetMagneticField() / aParticle.GetMagneticField().Magnitude();  // = b
    KThreeVector E1(e3.Z(), 0., -e3.X());
    KThreeVector e1 = E1 / E1.Magnitude();
    KThreeVector e2 = e3.Cross(e1);

    KThreeVector a = -1. * e1 * sin(aParticle.GetSpinAngle()) + e2 * cos(aParticle.GetSpinAngle());
    KThreeVector c = e1 * cos(aParticle.GetSpinAngle()) + e2 * sin(aParticle.GetSpinAngle());

    KThreeVector GradBMagnitude =
        aParticle.GetMagneticField() * aParticle.GetMagneticGradient() / aParticle.GetMagneticField().Magnitude();
    KThreeMatrix GradBDirection = aParticle.GetMagneticGradient() / aParticle.GetMagneticField().Magnitude() -
                                  KThreeMatrix::OuterProduct(aParticle.GetMagneticField(), GradBMagnitude) /
                                      aParticle.GetMagneticField().Magnitude() /
                                      aParticle.GetMagneticField().Magnitude();
    KThreeMatrix GradE1(GradBDirection[6],
                        0.,
                        -GradBDirection[0],
                        GradBDirection[7],
                        0.,
                        -GradBDirection[1],
                        GradBDirection[8],
                        0.,
                        -GradBDirection[2]);
    KThreeVector GradE1Magnitude = E1 * GradE1 / E1.Magnitude();
    KThreeMatrix Grade1 =
        GradE1 / E1.Magnitude() - KThreeMatrix::OuterProduct(E1, GradE1Magnitude) / E1.Magnitude() / E1.Magnitude();
    KThreeVector A = Grade1 * e2;

    KThreeVector tVelocity = fDirection * aParticle.GetVelocity();
    KThreeVector tForce =
        aParticle.GetCharge() * (aParticle.GetElectricField() + tVelocity.Cross(aParticle.GetMagneticField())) +
        aParticle.GetGyromagneticRatio() * aParticle.GetSpinMagnitude() * katrin::KConst::Hbar() *
            aParticle.GetAlignedSpin() * GradBMagnitude +
        aParticle.GetMagneticField().Magnitude() * aParticle.GetGyromagneticRatio() * aParticle.GetSpinMagnitude() *
            katrin::KConst::Hbar() * sqrt(1 - aParticle.GetAlignedSpin() * aParticle.GetAlignedSpin()) *
            (GradBDirection * c);
    double tMDot = sqrt(1 - aParticle.GetAlignedSpin() * aParticle.GetAlignedSpin()) * tVelocity * GradBDirection * c;
    double tPhiDot = -aParticle.GetGyromagneticRatio() * aParticle.GetMagneticField().Magnitude() -
                     aParticle.GetVelocity().Dot(A) -
                     aParticle.GetAlignedSpin() / sqrt(1 - aParticle.GetAlignedSpin() * aParticle.GetAlignedSpin()) *
                         (aParticle.GetVelocity() * GradBDirection * a);

    aDerivative.AddToVelocity(tVelocity);
    aDerivative.AddToForce(tForce);
    aDerivative.AddToMDot(tMDot);
    aDerivative.AddToPhiDot(tPhiDot);

    return;
}

void KSTrajTermPropagation::Differentiate(double /*aTime*/, const KSTrajMagneticParticle& aParticle,
                                          KSTrajMagneticDerivative& aDerivative) const
{
    KThreeVector tVelocity = fDirection * aParticle.GetMagneticField().Unit();

    aDerivative.AddToVelocity(tVelocity);

    return;
}

void KSTrajTermPropagation::Differentiate(double /*aTime*/, const KSTrajElectricParticle& aParticle,
                                          KSTrajElectricDerivative& aDerivative) const
{
    KThreeVector tVelocity = fDirection * aParticle.GetElectricField().Unit();

    aDerivative.AddToVelocity(tVelocity);

    return;
}

void KSTrajTermPropagation::SetDirection(const Direction& aDirection)
{
    fDirection = aDirection;
    return;
}

}  // namespace Kassiopeia
