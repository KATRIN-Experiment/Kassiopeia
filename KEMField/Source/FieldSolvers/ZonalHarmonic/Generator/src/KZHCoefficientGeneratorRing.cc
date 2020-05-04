#include "KZHCoefficientGeneratorRing.hh"

#include "KEMConstants.hh"
#include "KZHLegendreCoefficients.hh"

namespace KEMField
{
double KZHCoefficientGenerator<KRing>::Prefactor() const
{
    if (const auto* e = dynamic_cast<const KElectrostaticBasis*>(fRing))
        return e->GetSolution();
    else
        return 0.;
}

/**
   * Adds the contribution made due the ring to the central coefficents coeff.
   */
void KZHCoefficientGenerator<KRing>::ComputeCentralCoefficients(double z0, double rho, std::vector<double>& coeff) const
{
    double rho_ring = sqrt((z0 - fRing->GetZ()) * (z0 - fRing->GetZ()) + fRing->GetR() * fRing->GetR());
    double prefactor = Prefactor() / (4. * KEMConstants::Pi * KEMConstants::Eps0 * rho_ring);
    double rho_ratio = rho / rho_ring;

    std::vector<double> P1(coeff.size(), 0);  // vector of n legendre polynomials
    // P_n(cos(theta_ring))
    P1[0] = 1.;
    P1[1] = (fRing->GetZ() - z0) / rho_ring;  // cos(theta_ring)

    double Phi_cen = prefactor;

    coeff[0] += Phi_cen;
    Phi_cen *= rho_ratio;
    coeff[1] += Phi_cen * P1[1];

    for (unsigned int i = 2; i < coeff.size(); i++) {
        Phi_cen *= rho_ratio;
        P1[i] = (KZHLegendreCoefficients::GetInstance()->Get(0, i) * P1[1] * P1[i - 1] -
                 KZHLegendreCoefficients::GetInstance()->Get(1, i) * P1[i - 2]);

        coeff[i] += Phi_cen * P1[i];
    }
}

/**
   * Adds the contribution made due the coil to the remote coefficents coeff.
   */
void KZHCoefficientGenerator<KRing>::ComputeRemoteCoefficients(double z0, double rho, std::vector<double>& coeff) const
{
    double rho_ring = sqrt((z0 - fRing->GetZ()) * (z0 - fRing->GetZ()) + fRing->GetR() * fRing->GetR());
    double prefactor = Prefactor() / (4. * KEMConstants::Pi * KEMConstants::Eps0 * rho_ring);
    double rho_ratio = rho_ring / rho;

    std::vector<double> P1(coeff.size(), 0);  // vector of n legendre polynomials
    // P_n(cos(theta_ring))
    P1[0] = 1.;
    P1[1] = (fRing->GetZ() - z0) / rho_ring;  // cos(theta_ring)

    double Phi_rem = prefactor * rho_ratio;

    coeff[0] += Phi_rem;
    Phi_rem *= rho_ratio;
    coeff[1] += Phi_rem * P1[1];

    for (unsigned int i = 2; i < coeff.size(); i++) {
        Phi_rem *= rho_ratio;
        P1[i] = (KZHLegendreCoefficients::GetInstance()->Get(0, i) * P1[1] * P1[i - 1] -
                 KZHLegendreCoefficients::GetInstance()->Get(1, i) * P1[i - 2]);

        coeff[i] += Phi_rem * P1[i];
    }
}

/**
   * Computes rho_cen/rem for the ring corresponding to a source point located
   * at z0.  
   */
double KZHCoefficientGenerator<KRing>::ComputeRho(double z0, bool isCen) const
{
    double rho_ring = sqrt((fRing->GetZ() - z0) * (fRing->GetZ() - z0) + fRing->GetR() * fRing->GetR());

    if (isCen) {
        rho_ring -= fNullDistance;
        if (rho_ring < 0.)
            rho_ring = 0.;
    }
    else
        rho_ring += fNullDistance;

    return rho_ring;
}

void KZHCoefficientGenerator<KRing>::GetExtrema(double& zMin, double& zMax) const
{
    zMin = zMax = fRing->GetZ();
}
}  // namespace KEMField
