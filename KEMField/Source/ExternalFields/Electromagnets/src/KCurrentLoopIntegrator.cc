#include "KCurrentLoopIntegrator.hh"

#include "KEMConstants.hh"
#include "KEllipticIntegrals.hh"

#include <iomanip>

namespace KEMField
{
KThreeVector KCurrentLoopIntegrator::VectorPotential(const KCurrentLoop& currentLoop, const KPosition& P) const
{
    static KCompleteEllipticIntegral1stKind K_elliptic;
    static KEllipticEMinusKOverkSquared EK_elliptic;

    KPosition p = currentLoop.GetCoordinateSystem().ToLocal(P);

    double r = sqrt(p[0] * p[0] + p[1] * p[1]);

    double S = sqrt((currentLoop.GetP()[0] + r) * (currentLoop.GetP()[0] + r) +
                    (p[2] - currentLoop.GetP()[2]) * (p[2] - currentLoop.GetP()[2]));

    double k = 2. * sqrt(currentLoop.GetP()[0] * r) / S;

    double k_Elliptic = K_elliptic(k);
    double ek_Elliptic = EK_elliptic(k);

    double A_theta = -KEMConstants::Mu0OverPi * currentLoop.GetCurrent() * currentLoop.GetP()[0] / S *
                     (2. * ek_Elliptic + k_Elliptic);

    double sine = 0.;
    double cosine = 0.;

    if (r > 1.e-12) {
        cosine = p[0] / r;
        sine = p[1] / r;
    }

    return currentLoop.GetCoordinateSystem().ToGlobal(KThreeVector(-sine * A_theta, cosine * A_theta, 0.));
}

KThreeVector KCurrentLoopIntegrator::MagneticField(const KCurrentLoop& currentLoop, const KPosition& P) const
{
    static KCompleteEllipticIntegral1stKind K_elliptic;
    static KCompleteEllipticIntegral2ndKind E_elliptic;
    static KEllipticEMinusKOverkSquared EK_elliptic;

    KPosition p = currentLoop.GetCoordinateSystem().ToLocal(P);

    double r = sqrt(p[0] * p[0] + p[1] * p[1]);

    double S = sqrt((currentLoop.GetP()[0] + r) * (currentLoop.GetP()[0] + r) +
                    (p[2] - currentLoop.GetP()[2]) * (p[2] - currentLoop.GetP()[2]));

    double D = sqrt((currentLoop.GetP()[0] - r) * (currentLoop.GetP()[0] - r) +
                    (p[2] - currentLoop.GetP()[2]) * (p[2] - currentLoop.GetP()[2]));

    double k = 2. * sqrt(currentLoop.GetP()[0] * r) / S;

    double k_Elliptic = K_elliptic(k);
    double e_Elliptic = E_elliptic(k);

    double B_z = KEMConstants::Mu0OverPi * .5 * currentLoop.GetCurrent() / S *
                 (k_Elliptic + e_Elliptic * (2. * currentLoop.GetP()[0] * (currentLoop.GetP()[0] - r) / (D * D) - 1.));

    double B_r = 0;
    double cosine = 0;
    double sine = 0;

    if (r > 1.e-12) {
        double ek_Elliptic = EK_elliptic(k);

        B_r = KEMConstants::Mu0OverPi * currentLoop.GetCurrent() * (p[2] - currentLoop.GetP()[2]) *
              currentLoop.GetP()[0] / S * (2. / (S * S) * ek_Elliptic + e_Elliptic / (D * D));

        cosine = p[0] / r;
        sine = p[1] / r;
    }

    return currentLoop.GetCoordinateSystem().ToGlobal(KThreeVector(cosine * B_r, sine * B_r, B_z));
}
}  // namespace KEMField
