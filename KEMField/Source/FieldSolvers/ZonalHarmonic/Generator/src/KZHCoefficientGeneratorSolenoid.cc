#include "KZHCoefficientGeneratorSolenoid.hh"

#include "KEMConstants.hh"

#include "KZHLegendreCoefficients.hh"

namespace KEMField
{
  /**
   * Adds the contribution made due the ring to the central coefficents coeff.
   */
  void KZHCoefficientGenerator<KSolenoid>::ComputeCentralCoefficients(double z0,
							     double rho,
							     std::vector<double>& coeff) const
  {
    double rho_za = sqrt((z0-fSolenoid->GetZ0())*(z0-fSolenoid->GetZ0()) + fSolenoid->GetR()*fSolenoid->GetR());
    double rho_zb = sqrt((z0-fSolenoid->GetZ1())*(z0-fSolenoid->GetZ1()) + fSolenoid->GetR()*fSolenoid->GetR());
    double u_za = (fSolenoid->GetZ0()-z0)/rho_za; // cos(theta(Z_a))
    double u_zb = (fSolenoid->GetZ1()-z0)/rho_zb; // cos(theta(Z_b))

    double prefactor = -KEMConstants::Mu0*Prefactor()*.5;
    double rho_ratio_za = rho/rho_za;
    double rho_ratio_zb = rho/rho_zb;

    std::vector<double> P1p_za(coeff.size(),0); // P'_n(cos(theta(Z_a)))
    std::vector<double> P1p_zb(coeff.size(),0); // P'_n(cos(theta(Z_b)))

    P1p_za[0] = 0.;
    P1p_za[1] = 1.;

    P1p_zb[0] = 0.;
    P1p_zb[1] = 1.;

    coeff[0] += -prefactor*((fSolenoid->GetZ1() - z0)/sqrt(fSolenoid->GetR()*fSolenoid->GetR() + (fSolenoid->GetZ1()-z0)*(fSolenoid->GetZ1()-z0)) - (fSolenoid->GetZ0() - z0)/sqrt(fSolenoid->GetR()*fSolenoid->GetR() + (fSolenoid->GetZ0()-z0)*(fSolenoid->GetZ0()-z0)));

    //   double psi_cen_za = prefactor*(1.-u_za*u_za);
    //   double psi_cen_zb = prefactor*(1.-u_zb*u_zb);

    double psi_cen_za = prefactor*(1.-u_za*u_za);
    double psi_cen_zb = prefactor*(1.-u_zb*u_zb);

    for(unsigned int i=1;i<coeff.size()-1;i++)
    {
      P1p_za[i+1]=(KZHLegendreCoefficients::GetInstance()->Get(2,i+1)*u_za*P1p_za.at(i) -
		   KZHLegendreCoefficients::GetInstance()->Get(3,i+1)*P1p_za.at(i-1));
      P1p_zb[i+1]=(KZHLegendreCoefficients::GetInstance()->Get(2,i+1)*u_zb*P1p_zb.at(i) -
		   KZHLegendreCoefficients::GetInstance()->Get(3,i+1)*P1p_zb.at(i-1));

      psi_cen_za*=rho_ratio_za;
      psi_cen_zb*=rho_ratio_zb;
      coeff[i] += 1./((double)i)*(psi_cen_zb*P1p_zb[i] - psi_cen_za*P1p_za[i]);
    }
  }

  /**
   * Adds the contribution made due the coil to the remote coefficents coeff.
   */
  void KZHCoefficientGenerator<KSolenoid>::ComputeRemoteCoefficients(double z0,
							    double rho,
							    std::vector<double>& coeff) const
  {
    double rho_za = sqrt((z0-fSolenoid->GetZ0())*(z0-fSolenoid->GetZ0()) + fSolenoid->GetR()*fSolenoid->GetR());
    double rho_zb = sqrt((z0-fSolenoid->GetZ1())*(z0-fSolenoid->GetZ1()) + fSolenoid->GetR()*fSolenoid->GetR());
    double u_za = (fSolenoid->GetZ0()-z0)/rho_za; // cos(theta(Z_a))
    double u_zb = (fSolenoid->GetZ1()-z0)/rho_zb; // cos(theta(Z_b))

    double prefactor = KEMConstants::Mu0*Prefactor()*.5;
    double rho_ratio_za = rho_za/rho;
    double rho_ratio_zb = rho_zb/rho;

    std::vector<double> P1p_za(coeff.size(),0); // P'_n(cos(theta(Z_a)))
    std::vector<double> P1p_zb(coeff.size(),0); // P'_n(cos(theta(Z_b)))

    P1p_za[0] = 0.;
    P1p_za[1] = 1.;

    P1p_zb[0] = 0.;
    P1p_zb[1] = 1.;

    double psi_cen_za = prefactor*(1.-u_za*u_za)*rho_ratio_za;
    double psi_cen_zb = prefactor*(1.-u_zb*u_zb)*rho_ratio_zb;

    for(unsigned int i=1;i<coeff.size()-1;i++)
    {
      P1p_za[i+1]=(KZHLegendreCoefficients::GetInstance()->Get(2,i+1)*u_za*P1p_za.at(i) - KZHLegendreCoefficients::GetInstance()->Get(3,i+1)*P1p_za.at(i-1));
      P1p_zb[i+1]=(KZHLegendreCoefficients::GetInstance()->Get(2,i+1)*u_zb*P1p_zb.at(i) - KZHLegendreCoefficients::GetInstance()->Get(3,i+1)*P1p_zb.at(i-1));

      psi_cen_za*=rho_ratio_za;
      psi_cen_zb*=rho_ratio_zb;
      coeff[i] += 1./((double)(i+1))*(psi_cen_zb*P1p_zb[i] -
					psi_cen_za*P1p_za[i]);
    }
  }

  /**
   * Computes rho_cen/rem for the ring corresponding to a source point located
   * at z0.  
   */
  double KZHCoefficientGenerator<KSolenoid>::ComputeRho(double z0, bool isCen) const
  {
    double z = 0;

    if (isCen)
    {
      if (fabs(z0-fSolenoid->GetZ1())>fabs(z0-fSolenoid->GetZ0()))
	z = fSolenoid->GetZ0() - z0;
      else
	z = fSolenoid->GetZ1() - z0;
    }
    else
    {
      if (fabs(z0-fSolenoid->GetZ1())>fabs(z0-fSolenoid->GetZ0()))
	z = fSolenoid->GetZ1() - z0;
      else
	z = fSolenoid->GetZ0() - z0;
    }

    return sqrt(z*z+fSolenoid->GetR()*fSolenoid->GetR());
  }

  void KZHCoefficientGenerator<KSolenoid>::GetExtrema(double& zMin,double& zMax) const
  {
    zMin = fSolenoid->GetZ0(); zMax = fSolenoid->GetZ1();
  }
}
