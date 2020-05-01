#include "KSTrajTermSynchrotron.h"

#include "KConst.h"

namespace Kassiopeia
{

KSTrajTermSynchrotron::KSTrajTermSynchrotron() : fEnhancement(1.), fOldMethode(false) {}
KSTrajTermSynchrotron::KSTrajTermSynchrotron(const KSTrajTermSynchrotron& aCopy) :
    KSComponent(),
    fEnhancement(aCopy.fEnhancement),
    fOldMethode(aCopy.fOldMethode)
{}
KSTrajTermSynchrotron* KSTrajTermSynchrotron::Clone() const
{
    return new KSTrajTermSynchrotron(*this);
}
KSTrajTermSynchrotron::~KSTrajTermSynchrotron() {}

void KSTrajTermSynchrotron::Differentiate(double /*aTime*/, const KSTrajExactParticle& aParticle,
                                          KSTrajExactDerivative& aDerivative) const
{
    double Q = aParticle.GetCharge();
    double M = aParticle.GetMass();
    double P = aParticle.GetMomentum().Magnitude();
    double Gamma = aParticle.GetLorentzFactor();
    double C = katrin::KConst::C();
    if (!fOldMethode) {
        //New Methode: Equations from abraham lorentz force
        //the energy loss leads to an smaller polar angle, as mostly the orthogonal momentum is reduced.
        KThreeVector tVelocity = aParticle.GetVelocity();
        KThreeVector tLorentzForce = Q * (aParticle.GetElectricField() + tVelocity.Cross(aParticle.GetMagneticField()));

        KThreeVector tAcceleration = tLorentzForce / (M * Gamma) - 1.0 / (M * M * C * C * Gamma * Gamma) * tVelocity *
                                                                       aParticle.GetMomentum().Dot(tLorentzForce);
        double tTau = Q * Q / (6.0 * katrin::KConst::Pi() * katrin::KConst::EpsNull() * M * C * C * C);

        KThreeMatrix tElectricFieldGradient = aParticle.GetElectricGradient();
        KThreeVector tMagneticField = aParticle.GetMagneticField();
        KThreeMatrix tMagneticFieldGradient = aParticle.GetMagneticGradient();

        KThreeVector tFirstTerm = tVelocity * tElectricFieldGradient + tAcceleration.Cross(tMagneticField) +
                                  tVelocity.Cross(tVelocity * tMagneticFieldGradient);

        KThreeVector tSecondTerm =
            Gamma * Gamma * Gamma / (C * C) * (tAcceleration.Cross(tVelocity.Cross(tLorentzForce)));

        KThreeVector tForce = fEnhancement * tTau * (Gamma * Q * tFirstTerm - tSecondTerm);

        aDerivative.AddToForce(tForce);
        fTotalForce = tForce.Magnitude();
    }
    else {
        //Old Methode: Equations from Dans derivations
        //correct energy loss, but the total energy is reduced, not the orthogonal component
        double Factor = (katrin::KConst::MuNull() / (6. * katrin::KConst::Pi() * katrin::KConst::C())) *
                        ((Q * Q * Q * Q) / (M * P * P));

        KThreeVector tTUnit = aParticle.GetMomentum().Unit();
        KThreeVector tUUnit = aParticle.GetMomentum().Cross(aParticle.GetMagneticField()).Unit();
        KThreeVector tVUnit = tTUnit.Cross(tUUnit).Unit();

        double tET = aParticle.GetElectricField().Dot(tTUnit);
        double tEU = aParticle.GetElectricField().Dot(tUUnit);
        double tEV = aParticle.GetElectricField().Dot(tVUnit);
        double tBV = aParticle.GetMagneticField().Dot(tVUnit);
        double tXi1 = tET * tET + tBV * tBV * (P / M) * (P / M);
        double tXi2 = -2. * tEU * tBV * (P / M);
        double tXi3 = tEU * tEU + tEV * tEV;

        KThreeVector tForce = -fEnhancement * Factor *
                              (Gamma * tXi1 + Gamma * Gamma * tXi2 + Gamma * Gamma * Gamma * tXi3) *
                              aParticle.GetMomentum();

        aDerivative.AddToForce(tForce);
        fTotalForce = tForce.Magnitude();
    }

    return;
}
void KSTrajTermSynchrotron::Differentiate(double /*aTime*/, const KSTrajAdiabaticParticle& aParticle,
                                          KSTrajAdiabaticDerivative& aDerivative) const
{
    double Q = aParticle.GetCharge();
    double M = aParticle.GetMass();
    double Factor = (katrin::KConst::MuNull() / (6. * katrin::KConst::Pi() * katrin::KConst::C())) *
                    ((Q * Q * Q * Q) / (M * M * M));

    double tForce = -fEnhancement * Factor * aParticle.GetLorentzFactor() *
                    aParticle.GetMagneticField().MagnitudeSquared() * aParticle.GetTransMomentum();

    aDerivative.AddToTransverseForce(tForce);
    fTotalForce = tForce;

    return;
}
void KSTrajTermSynchrotron::Differentiate(double aTime, const KSTrajExactTrappedParticle& aParticle,
                                          KSTrajExactTrappedDerivative& aDerivative) const
{
    if (aTime == 0)
        return;

    double Q = aParticle.GetCharge();
    double M = aParticle.GetMass();
    double P = aParticle.GetMomentum().Magnitude();
    double Gamma = aParticle.GetLorentzFactor();
    double C = katrin::KConst::C();
    if (!fOldMethode) {
        //New Methode: Equations from abraham lorentz force
        //the energy loss leads to an smaller polar angle, as mostly the orthogonal momentum is reduced.
        KThreeVector tVelocity = aParticle.GetVelocity();
        KThreeVector tLorentzForce = Q * (aParticle.GetElectricField() + tVelocity.Cross(aParticle.GetMagneticField()));

        KThreeVector tAcceleration = tLorentzForce / (M * Gamma) - 1.0 / (M * M * C * C * Gamma * Gamma) * tVelocity *
                                                                       aParticle.GetMomentum().Dot(tLorentzForce);
        double tTau = Q * Q / (6.0 * katrin::KConst::Pi() * katrin::KConst::EpsNull() * M * C * C * C);

        KThreeMatrix tElectricFieldGradient = aParticle.GetElectricGradient();
        KThreeVector tMagneticField = aParticle.GetMagneticField();
        KThreeMatrix tMagneticFieldGradient = aParticle.GetMagneticGradient();

        KThreeVector tFirstTerm = tVelocity * tElectricFieldGradient + tAcceleration.Cross(tMagneticField) +
                                  tVelocity.Cross(tVelocity * tMagneticFieldGradient);

        KThreeVector tSecondTerm =
            Gamma * Gamma * Gamma / (C * C) * (tAcceleration.Cross(tVelocity.Cross(tLorentzForce)));

        KThreeVector tForce = fEnhancement * tTau * (Gamma * Q * tFirstTerm - tSecondTerm);

        aDerivative.AddToForce(tForce);
        fTotalForce = tForce.Magnitude();
    }
    else {
        //Old Methode: Equations from Dans derivations
        //correct energy loss, but the total energy is reduced, not the orthogonal component
        double Factor = (katrin::KConst::MuNull() / (6. * katrin::KConst::Pi() * katrin::KConst::C())) *
                        ((Q * Q * Q * Q) / (M * P * P));

        KThreeVector tTUnit = aParticle.GetMomentum().Unit();
        KThreeVector tUUnit = aParticle.GetMomentum().Cross(aParticle.GetMagneticField()).Unit();
        KThreeVector tVUnit = tTUnit.Cross(tUUnit).Unit();

        double tET = aParticle.GetElectricField().Dot(tTUnit);
        double tEU = aParticle.GetElectricField().Dot(tUUnit);
        double tEV = aParticle.GetElectricField().Dot(tVUnit);
        double tBV = aParticle.GetMagneticField().Dot(tVUnit);
        double tXi1 = tET * tET + tBV * tBV * (P / M) * (P / M);
        double tXi2 = -2. * tEU * tBV * (P / M);
        double tXi3 = tEU * tEU + tEV * tEV;

        KThreeVector tForce = -fEnhancement * Factor *
                              (Gamma * tXi1 + Gamma * Gamma * tXi2 + Gamma * Gamma * Gamma * tXi3) *
                              aParticle.GetMomentum();

        aDerivative.AddToForce(tForce);
        fTotalForce = tForce.Magnitude();
    }

    return;
}

void KSTrajTermSynchrotron::SetEnhancement(const double& anEnhancement)
{
    fEnhancement = anEnhancement;
    return;
}

void KSTrajTermSynchrotron::SetOldMethode(const bool& aBool)
{
    fOldMethode = aBool;
    return;
}

STATICINT sKSTrajTermSynchrotronDict =
    KSDictionary<KSTrajTermSynchrotron>::AddComponent(&KSTrajTermSynchrotron::GetTotalForce, "total_force");
}  // namespace Kassiopeia
