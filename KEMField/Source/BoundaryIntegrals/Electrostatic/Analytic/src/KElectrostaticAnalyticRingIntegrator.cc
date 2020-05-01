#include "KElectrostaticAnalyticRingIntegrator.hh"

#include "KEllipticIntegrals.hh"
#include "KGaussianQuadrature.hh"

#include <iomanip>

namespace KEMField
{
/**
 * \image html potentialFromRing.gif
 * Returns the electric potential at a point P (P[0],P[1],P[2]) due to the ring
 * by computing the following integral:
 * \f{eqnarray*}{
 * V(\vec{p}) &=& \frac{1}{4 \pi \epsilon_0} \int_{0}^{2 \pi} \frac{\lambda R \cdot d\theta}{\sqrt{R^2+r^2+(z-Z)^2-2Rr \cos \theta}}=\\
 * &=& \frac{Q}{2 \pi^2 \epsilon_0}\cdot \frac{K(k)}{S},
 * \f}
 * where
 * \f{eqnarray*}{
 * r &=& \sqrt{(P[0])^2+(P[1])^2},\\
 * z &=& P[2], \\
 * S &=& \sqrt{(R+r)^2+(z-Z)^2},\\
 * k &=& \frac{2\sqrt{Rr}}{S}.
 * \f}
 */
double KElectrostaticAnalyticRingIntegrator::Potential(const KRing* source, const KPosition& P) const
{
    double par[7];

    par[0] = P[2];
    par[1] = sqrt((P[0] * P[0]) + (P[1] * P[1]));
    par[2] = par[4] = source->GetZ();
    par[3] = par[5] = source->GetR();
    par[6] = 1.;

    double c = 1. / (par[3] * 2. * KEMConstants::Pi * KEMConstants::Pi * KEMConstants::Eps0);

    return c * PotentialFromChargedRing(P, par);
}


KThreeVector KElectrostaticAnalyticRingIntegrator::ElectricField(const KRing* source, const KPosition& P) const
{
    double par[7];

    par[0] = P[2];
    par[1] = sqrt((P[0] * P[0]) + (P[1] * P[1]));
    par[2] = par[4] = source->GetZ();
    par[3] = par[5] = source->GetR();
    par[6] = 1.;

    double c = 1. / (par[3] * 2. * KEMConstants::Pi * KEMConstants::Pi * KEMConstants::Eps0);

    KThreeVector field;

    field[2] = c * EFieldZFromChargedRing(P, par);
    double Er = c * EFieldRFromChargedRing(P, par);

    if (par[1] < 1.e-14)
        field[0] = field[1] = 0;
    else {
        double cosine = P[0] / par[1];
        double sine = P[1] / par[1];

        field[0] = cosine * Er;
        field[1] = sine * Er;
    }
    return field;
}

/**
 * computes electric potential at point (r,z) due to an infinitely thin
 * charged ring with axial coordinate Z and radius R.  The input parameters
 * for this function are defined in order to facilitate integration over p,
 * the distance between (rA,zA) and a point between (rA,zA) and (rB,zB), the
 * endpoints of an electrode.
 *
 * \f[
 * \Phi = \left( \frac{q}{4 R \pi^2 \epsilon_0} \right) \cdot \frac{R\cdot K}{S}
 * \f]
 *
 * NOTE: the prefactor \f$\left( \frac{q}{4 R \pi^2 \epsilon_0} \right)\f$ is
 *       included after integrating.
 * @code
 *     - z    = par[0]
 *     - r    = par[1]
 *     - zA   = par[2]
 *     - rA   = par[3]
 *     - zB   = par[4]
 *     - rB   = par[5]
 *     - L    = par[6]
 *
 *     - Z(p) = zA+p/L*(zB-zA)
 *     - R(p) = rA+p/L*(rB-rA)
 * @endcode
 */
double KElectrostaticAnalyticRingIntegrator::PotentialFromChargedRing(const double* P, double* par)
{
    static KCompleteEllipticIntegral1stKind K_elliptic;

    double Z = par[2] + P[0] / par[6] * (par[4] - par[2]);
    double R = par[3] + P[0] / par[6] * (par[5] - par[3]);

    double dz = par[0] - Z;
    double dr = par[1] - R;
    double sumr = R + par[1];

    double eta = (dr * dr + dz * dz) / (sumr * sumr + dz * dz);
    double k = (eta > 1. ? 0. : sqrt(1. - eta));
    double K = K_elliptic(k);

    double S = sqrt(sumr * sumr + dz * dz);

    return R * K / S;
}

/**
 * computes R-component of E-Field at point (r,z) due to an infinitely thin
 * charged ring with axial coordinate Z and radius R.  The input parameters
 * for this function are defined in order to facilitate integration over p,
 * the distance between (rA,zA) and a point between (rA,zA) and (rB,zB), the
 * endpoints of an electrode.
 *
 * \f[
 * E_r = \left( \frac{q}{\pi \epsilon_0} \right) \cdot \frac{R}{S^3} \cdot \left( -2\cdot R \cdot EK+\frac{(r-R}{\eta \cdot E} \right)
 * \f]
 *
 * NOTE: the prefactor \f$\left( \frac{q}{\pi \epsilon_0} \right)\f$ is
 *       included after integrating.
 * @code
 *     - z    = par[0]
 *     - r    = par[1]
 *     - zA   = par[2]
 *     - rA   = par[3]
 *     - zB   = par[4]
 *     - rB   = par[5]
 *     - L    = par[6]
 *
 *     - Z(p) = zA+p/L*(zB-zA)
 *     - R(p) = rA+p/L*(rB-rA)
 * @endcode
 */
double KElectrostaticAnalyticRingIntegrator::EFieldRFromChargedRing(const double* P, double* par)
{
    static KCompleteEllipticIntegral2ndKind E_elliptic;
    static KEllipticEMinusKOverkSquared EK_elliptic;

    double Z = par[2] + P[0] / par[6] * (par[4] - par[2]);
    double R = par[3] + P[0] / par[6] * (par[5] - par[3]);

    double dz = par[0] - Z;
    double dr = par[1] - R;
    double sumr = R + par[1];

    double eta = (dr * dr + dz * dz) / (sumr * sumr + dz * dz);
    double k = sqrt(1. - eta);
    double E = E_elliptic(k);
    double S = sqrt(sumr * sumr + dz * dz);
    double EK = EK_elliptic(k);

    return R / (S * S * S) * (-2. * R * EK + dr / eta * E);
}

/**
 * computes Z-component of E-Field at point (r,z) due to an infinitely thin
 * charged ring with axial coordinate Z and radius R.  The input parameters
 * for this function are defined in order to facilitate integration over p,
 * the distance between (rA,zA) and a point between (rA,zA) and (rB,zB), the
 * endpoints of an electrode.
 *
 *  EFieldZ = q/(Pi*e_0) * (z-Z)/S^3*E/eta*R : the prefactor is included after
 *                                             integrating
 * \f[
 * E_z = \left( \frac{q}{\pi \epsilon_0} \right) \cdot \left( \frac{(z-Z)}{S^3}\cdot \frac{E}{\eta \cdot R} \right)
 * \f]
 *
 * NOTE: the prefactor \f$\left( \frac{q}{\pi \epsilon_0} \right)\f$ is
 *       included after integrating.
 * @code
 *     - z    = par[0]
 *     - r    = par[1]
 *     - zA   = par[2]
 *     - rA   = par[3]
 *     - zB   = par[4]
 *     - rB   = par[5]
 *     - L    = par[6]
 *
 *     - Z(p) = zA+p/L*(zB-zA)
 *     - R(p) = rA+p/L*(rB-rA)
 * @endcode
 */
double KElectrostaticAnalyticRingIntegrator::EFieldZFromChargedRing(const double* P, double* par)
{
    static KCompleteEllipticIntegral2ndKind E_elliptic;
    double Z = par[2] + P[0] / par[6] * (par[4] - par[2]);
    double R = par[3] + P[0] / par[6] * (par[5] - par[3]);

    double dz = par[0] - Z;
    double dr = par[1] - R;
    double sumr = R + par[1];

    double eta = (dr * dr + dz * dz) / (sumr * sumr + dz * dz);
    double k = sqrt(1. - eta);
    double E = E_elliptic(k);
    double S = sqrt(sumr * sumr + dz * dz);

    return dz / (S * S * S) * E / eta * R;
}

}  // namespace KEMField
