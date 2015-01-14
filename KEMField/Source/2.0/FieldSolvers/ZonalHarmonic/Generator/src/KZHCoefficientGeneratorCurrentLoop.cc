#include "KZHCoefficientGeneratorCurrentLoop.hh"

#include "KEMConstants.hh"

#include "KZHLegendreCoefficients.hh"

namespace KEMField
{
  /**
   * Adds the contribution made due the ring to the central coefficents coeff.
   */
  void KZHCoefficientGenerator<KCurrentLoop>::ComputeCentralCoefficients(double z0,
								double rho,
								std::vector<double>& coeff) const
  {
    double rho_ring = sqrt((z0-fCurrentLoop->GetZ())*(z0-fCurrentLoop->GetZ()) + fCurrentLoop->GetR()*fCurrentLoop->GetR());
    double u = (fCurrentLoop->GetZ()-z0)/rho_ring; // cos(theta_ring)
    double prefactor = KEMConstants::Mu0*Prefactor()*.5*(1.-u*u)/rho;
    double rho_ratio = rho/rho_ring;

    std::vector<double> P1p(coeff.size(),0); // P'_n(cos(theta_ring))

    P1p[0] = 0.;
    P1p[1] = 1.;

    double Psi_cen = prefactor*rho_ratio;

    coeff[0] += Psi_cen;

    for(unsigned int i=1;i<coeff.size()-1;i++)
    {
      P1p[i+1]=(KZHLegendreCoefficients::GetInstance()->Get(2,i+1)*u*P1p.at(i) -
  		KZHLegendreCoefficients::GetInstance()->Get(3,i+1)*P1p.at(i-1));

      Psi_cen*=rho_ratio;
      coeff[i] += Psi_cen*P1p.at(i+1);
    }
  }

  /**
   * Adds the contribution made due the coil to the remote coefficents coeff.
   */
  void KZHCoefficientGenerator<KCurrentLoop>::ComputeRemoteCoefficients(double z0,
							       double rho,
							       std::vector<double>& coeff) const
  {
    double rho_ring = sqrt((z0-fCurrentLoop->GetZ())*(z0-fCurrentLoop->GetZ()) + fCurrentLoop->GetR()*fCurrentLoop->GetR());
    double u = (fCurrentLoop->GetZ()-z0)/rho_ring; // cos(theta_ring)
    double prefactor = KEMConstants::Mu0*Prefactor()*.5*(1.-u*u)/rho;
    double rho_ratio = rho_ring/rho;

    std::vector<double> P1p(coeff.size(),0); // P'_n(cos(theta_ring))

    P1p[0] = 0.;
    P1p[1] = 1.;

    double Psi_rem = prefactor*rho_ratio;

    for (unsigned int i=2;i<coeff.size();i++)
    {
      P1p[i]=(KZHLegendreCoefficients::GetInstance()->Get(2,i)*u*P1p.at(i-1) -
  	      KZHLegendreCoefficients::GetInstance()->Get(3,i)*P1p.at(i-2));

      Psi_rem*=rho_ratio;
      coeff[i] += Psi_rem*P1p.at(i-1);
    }
  }

  /**
   * Computes rho_cen/rem for the ring corresponding to a source point located
   * at z0.  
   */
  double KZHCoefficientGenerator<KCurrentLoop>::ComputeRho(double z0, bool) const
  {
    return sqrt((fCurrentLoop->GetZ()-z0)*(fCurrentLoop->GetZ()-z0) + fCurrentLoop->GetR()*fCurrentLoop->GetR());
  }

  void KZHCoefficientGenerator<KCurrentLoop>::GetExtrema(double& zMin,double& zMax) const
  {
    zMin = zMax = fCurrentLoop->GetZ();
  }
}
