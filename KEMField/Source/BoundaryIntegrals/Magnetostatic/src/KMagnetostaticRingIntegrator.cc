#include "KMagnetostaticRingIntegrator.hh"

#include "KEllipticIntegrals.hh"

#include <iomanip>

namespace KEMField
{
KFieldVector KMagnetostaticRingIntegrator::VectorPotential(const KRing* source, const KPosition& P) const
{
    static KCompleteEllipticIntegral1stKind K_elliptic;
    static KEllipticEMinusKOverkSquared EK_elliptic;

    double r = sqrt(P[0] * P[0] + P[1] * P[1]);

    double S = sqrt((source->GetP()[0] + r) * (source->GetP()[0] + r) +
                    (P[2] - source->GetP()[2]) * (P[2] - source->GetP()[2]));

    double k = 2. * sqrt(source->GetP()[0] * r) / S;

    double k_Elliptic = K_elliptic(k);
    double ek_Elliptic = EK_elliptic(k);

    double A_theta = -KEMConstants::Mu0OverPi * source->GetP()[0] / S * (2. * ek_Elliptic + k_Elliptic);

    double sine = 0.;
    double cosine = 0.;

    if (r > 1.e-12) {
        cosine = P[0] / r;
        sine = P[1] / r;
    }

    return KFieldVector(-sine * A_theta, cosine * A_theta, 0.);
}

KFieldVector KMagnetostaticRingIntegrator::MagneticField(const KRing* source, const KPosition& P) const
{
    static KCompleteEllipticIntegral1stKind K_elliptic;
    static KCompleteEllipticIntegral2ndKind E_elliptic;
    static KEllipticEMinusKOverkSquared EK_elliptic;

    double r = sqrt(P[0] * P[0] + P[1] * P[1]);

    double S = sqrt((source->GetP()[0] + r) * (source->GetP()[0] + r) +
                    (P[2] - source->GetP()[2]) * (P[2] - source->GetP()[2]));

    double D = sqrt((source->GetP()[0] - r) * (source->GetP()[0] - r) +
                    (P[2] - source->GetP()[2]) * (P[2] - source->GetP()[2]));

    double k = 2. * sqrt(source->GetP()[0] * r) / S;

    double k_Elliptic = K_elliptic(k);
    double e_Elliptic = E_elliptic(k);

    double B_z = KEMConstants::Mu0OverPi * .5 * source->GetCurrent() / S *
                 (k_Elliptic + e_Elliptic * (2. * source->GetP()[0] * (source->GetP()[0] - r) / (D * D) - 1.));

    double B_r = 0;
    double cosine = 0;
    double sine = 0;

    if (r > 1.e-12) {
        double ek_Elliptic = EK_elliptic(k);

        B_r = KEMConstants::Mu0OverPi * (P[2] - source->GetP()[2]) * source->GetP()[0] / S *
              (2. / (S * S) * ek_Elliptic + e_Elliptic / (D * D));

        cosine = P[0] / r;
        sine = P[1] / r;
    }

    return KFieldVector(cosine * B_r, sine * B_r, B_z);
}

KFieldVector KMagnetostaticRingIntegrator::VectorPotential(const KSymmetryGroup<KRing>* source,
                                                           const KPosition& P) const
{
    KFieldVector A;
    for (KSymmetryGroup<KRing>::ShapeCIt it = source->begin(); it != source->end(); ++it)
        A += VectorPotential(*it, P);
    return A;
}

KFieldVector KMagnetostaticRingIntegrator::MagneticField(const KSymmetryGroup<KRing>* source, const KPosition& P) const
{
    KFieldVector magneticField(0., 0., 0.);
    for (KSymmetryGroup<KRing>::ShapeCIt it = source->begin(); it != source->end(); ++it)
        magneticField += MagneticField(*it, P);
    return magneticField;
}
}  // namespace KEMField
