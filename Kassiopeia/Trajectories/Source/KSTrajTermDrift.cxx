#include "KSTrajTermDrift.h"


namespace Kassiopeia
{

KSTrajTermDrift::KSTrajTermDrift() = default;
KSTrajTermDrift::KSTrajTermDrift(const KSTrajTermDrift&) : KSComponent() {}
KSTrajTermDrift* KSTrajTermDrift::Clone() const
{
    return new KSTrajTermDrift(*this);
}
KSTrajTermDrift::~KSTrajTermDrift() = default;

void KSTrajTermDrift::Differentiate(double /*aTime*/, const KSTrajAdiabaticParticle& aParticle,
                                    KSTrajAdiabaticDerivative& aDerivative) const
{
    KGeoBag::KThreeVector tMagneticField = aParticle.GetMagneticField();
    KGeoBag::KThreeVector tMagneticFieldUnit = tMagneticField.Unit();
    KGeoBag::KThreeMatrix tMagneticGradient = aParticle.GetMagneticGradient();
    KGeoBag::KThreeVector tMagneticGradientUnit;
    tMagneticGradientUnit.X() = tMagneticFieldUnit.X() * tMagneticGradient(0, 0) +
                                tMagneticFieldUnit.Y() * tMagneticGradient(0, 1) +
                                tMagneticFieldUnit.Z() * tMagneticGradient(0, 2);
    tMagneticGradientUnit.Y() = tMagneticFieldUnit.X() * tMagneticGradient(1, 0) +
                                tMagneticFieldUnit.Y() * tMagneticGradient(1, 1) +
                                tMagneticFieldUnit.Z() * tMagneticGradient(1, 2);
    tMagneticGradientUnit.Z() = tMagneticFieldUnit.X() * tMagneticGradient(2, 0) +
                                tMagneticFieldUnit.Y() * tMagneticGradient(2, 1) +
                                tMagneticFieldUnit.Z() * tMagneticGradient(2, 2);

    KGeoBag::KThreeVector tElectricField = aParticle.GetElectricField();
    double tMagneticFieldMag = tMagneticField.Magnitude();
    double tMagneticFieldMag2 = tMagneticField.MagnitudeSquared();
    double tMagneticFieldMag3 = tMagneticFieldMag2 * tMagneticFieldMag;
    double tLongMomentum = aParticle.GetLongMomentum();
    double tLongMomentum2 = tLongMomentum * tLongMomentum;
    double tLorentzFactor = aParticle.GetLorentzFactor();
    double tTransMomentum = aParticle.GetTransMomentum();
    double tTransMomentum2 = tTransMomentum * tTransMomentum;
    double tMass = aParticle.GetMass();
    double tCharge = aParticle.GetCharge();

    KGeoBag::KThreeVector tDriftVelocity =
        (1. / tMagneticFieldMag2) * tElectricField.Cross(tMagneticField) +
        ((2. * tLongMomentum2 + tTransMomentum2) / (tCharge * tMagneticFieldMag3 * tMass * (1. + tLorentzFactor))) *
            (tMagneticField.Cross(tMagneticGradientUnit));
    double tLongitudinalForce =
        ((-1. * tTransMomentum2) / (2. * tMagneticFieldMag * tLongMomentum)) *
            tMagneticFieldUnit.Dot(tMagneticGradient * tDriftVelocity) +
        ((tCharge * tLorentzFactor * tMass) / (tLongMomentum)) * tElectricField.Dot(tDriftVelocity);
    double tTransverseForce =
        ((tTransMomentum) / (2. * tMagneticFieldMag)) * tMagneticFieldUnit.Dot(tMagneticGradient * tDriftVelocity);

    trajmsg_debug("adiabatic drift gc velocity: <" << tDriftVelocity << ">" << ret)
        trajmsg_debug("adiabatic drift longitudinal force <" << tLongitudinalForce << ">" << ret)
            trajmsg_debug("adiabatic drift transverse force <" << tTransverseForce << ">" << ret)

                aDerivative.AddToGuidingCenterVelocity(tDriftVelocity);
    aDerivative.AddToLongitudinalForce(tLongitudinalForce);
    aDerivative.AddToTransverseForce(tTransverseForce);
    fDriftVelocity = tDriftVelocity;
    fLongitudinalForce = tLongitudinalForce;
    fTransverseForce = tTransverseForce;

    return;
}

STATICINT sKSTrajTermDriftDict =
    KSDictionary<KSTrajTermDrift>::AddComponent(&KSTrajTermDrift::GetDriftVelocity, "gc_velocity") +
    KSDictionary<KSTrajTermDrift>::AddComponent(&KSTrajTermDrift::GetLongitudinalForce, "longitudinal_force") +
    KSDictionary<KSTrajTermDrift>::AddComponent(&KSTrajTermDrift::GetTransverseForce, "transverse_force");
}  // namespace Kassiopeia
