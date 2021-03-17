#include "KSolenoidIntegrator.hh"

#include "KEMConstants.hh"
#include "KEllipticIntegrals.hh"

#include <limits>

namespace KEMField
{
KFieldVector KSolenoidIntegrator::VectorPotential(const KSolenoid& solenoid, const KPosition& P) const
{
    KPosition localP = solenoid.GetCoordinateSystem().ToLocal(P);
    double p[1] = {solenoid.GetP0()[0]};
    double par[4] = {localP[2],                                            // z
                     sqrt(localP[0] * localP[0] + localP[1] * localP[1]),  // r
                     solenoid.GetP0()[2],                                  // z_min
                     solenoid.GetP1()[2]};                                 // z_max

    double a_theta = KEMConstants::Mu0OverPi * solenoid.GetCurrentDensity() * A_theta(p, par);

    double cosine = 0;
    double sine = 0;

    if (par[1] > 1.e-15) {
        cosine = localP[0] / par[1];
        sine = localP[1] / par[1];
    }

    return solenoid.GetCoordinateSystem().ToGlobal(KFieldVector(-sine * a_theta, cosine * a_theta, 0.));
}

KFieldVector KSolenoidIntegrator::MagneticField(const KSolenoid& solenoid, const KPosition& P) const
{
    KPosition localP = solenoid.GetCoordinateSystem().ToLocal(P);
    double p[1] = {solenoid.GetP0()[0]};
    double par[4] = {localP[2],                                            // z
                     sqrt(localP[0] * localP[0] + localP[1] * localP[1]),  // r
                     solenoid.GetP0()[2],                                  // z_min
                     solenoid.GetP1()[2]};                                 // z_max

    double b_z = -KEMConstants::Mu0OverPi * solenoid.GetCurrentDensity() * B_z(p, par);

    double b_r = 0;
    double cosine = 0;
    double sine = 0;

    if (par[1] > 1.e-12) {
        b_r = -KEMConstants::Mu0OverPi * solenoid.GetCurrentDensity() * B_r(p, par);

        cosine = localP[0] / par[1];
        sine = localP[1] / par[1];
    }

    return solenoid.GetCoordinateSystem().ToGlobal(KFieldVector(cosine * b_r, sine * b_r, b_z));
}

double KSolenoidIntegrator::A_theta(const double* p, const double* par)
{
    static KEllipticCarlsonSymmetricRD ellipticCarlsonSymmetricRD;
    static KEllipticCarlsonSymmetricRJ ellipticCarlsonSymmetricRJ;
    static const double lolim = pow(5.0 * std::numeric_limits<double>::min(), 1.0 / 3.0);

    double A_theta[2];

    double dr = par[1] - p[0];
    double sumr = par[1] + p[0];

    for (int i = 0; i < 2; i++) {
        double dz = par[0] - par[i + 2];
        double S = sqrt(sumr * sumr + dz * dz);

        double eta_1 = (dr * dr + dz * dz) / (sumr * sumr + dz * dz);
        double eta_2 = (dr * dr) / (sumr * sumr);

        if (eta_1 < lolim)
            eta_1 = lolim;
        if (eta_2 < lolim)
            eta_2 = lolim;

        A_theta[i] =
            dz * p[0] / (S * 3.) *
            (eta_2 * ellipticCarlsonSymmetricRJ(0., eta_1, 1., eta_2) - ellipticCarlsonSymmetricRD(0., eta_1, 1.));
    }

    return (A_theta[1] - A_theta[0]);
}

double KSolenoidIntegrator::B_r(const double* p, const double* par)
{
    static KEllipticCarlsonSymmetricRF ellipticCarlsonSymmetricRF;
    static KEllipticCarlsonSymmetricRD ellipticCarlsonSymmetricRD;
    const double lolim = 2.0 / pow(std::numeric_limits<double>::max(), 2.0 / 3.0);

    double Br[2];

    double dr = par[1] - p[0];
    double sumr = par[1] + p[0];

    for (int i = 0; i < 2; i++) {
        double dz = par[0] - par[i + 2];
        double S = sqrt(sumr * sumr + dz * dz);

        double eta = (dr * dr + dz * dz) / (sumr * sumr + dz * dz);
        if (eta < lolim)
            eta = lolim;

        Br[i] =
            p[0] / S * (ellipticCarlsonSymmetricRF(0., eta, 1.) - 2. * ellipticCarlsonSymmetricRD(0., eta, 1.) / 3.);
    }

    return (Br[1] - Br[0]);
}

double KSolenoidIntegrator::B_z(const double* p, const double* par)
{
    static KEllipticCarlsonSymmetricRF ellipticCarlsonSymmetricRF;
    static KEllipticCarlsonSymmetricRJ ellipticCarlsonSymmetricRJ;
    static const double lolim = pow(5.0 * std::numeric_limits<double>::min(), 1.0 / 3.0);

    double Bz[2];

    double dr = par[1] - p[0];
    double sumr = par[1] + p[0];

    for (int i = 0; i < 2; i++) {
        double dz = par[0] - par[i + 2];
        double S = sqrt(sumr * sumr + dz * dz);

        double eta_1 = (dr * dr + dz * dz) / (sumr * sumr + dz * dz);
        double eta_2 = (dr * dr) / (sumr * sumr);

        if (eta_1 < lolim)
            eta_1 = lolim;
        if (eta_2 < lolim)
            eta_2 = lolim;

        Bz[i] = dz * p[0] / (S * sumr) *
                (ellipticCarlsonSymmetricRF(0., eta_1, 1.) -
                 dr * (1. - eta_2) / (6. * p[0]) * ellipticCarlsonSymmetricRJ(0., eta_1, 1., eta_2));
    }
    return (Bz[1] - Bz[0]);
}
}  // namespace KEMField
