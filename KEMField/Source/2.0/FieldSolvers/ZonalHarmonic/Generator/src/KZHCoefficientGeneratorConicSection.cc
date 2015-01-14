#include "KZHCoefficientGeneratorConicSection.hh"

#include "KEMConstants.hh"

#include "KZHLegendreCoefficients.hh"

namespace KEMField
{
  double KZHCoefficientGenerator<KConicSection>::Prefactor() const
  {
    if (const KElectrostaticBasis* e =
	dynamic_cast<const KElectrostaticBasis*>(fConicSection))
      return e->GetSolution();
    else
      return 0.;
  }

  /**
   * Adds the contribution made due the ring to the central coefficents coeff.
   */
  void KZHCoefficientGenerator<KConicSection>
  ::ComputeCentralCoefficients(double z0,double rho,std::vector<double>& coeff) const
  {
    ComputeCoefficients(z0,rho,coeff,true);
  }

  /**
   * Adds the contribution made due the coil to the remote coefficents coeff.
   */
  void KZHCoefficientGenerator<KConicSection>
  ::ComputeRemoteCoefficients(double z0,double rho,std::vector<double>& coeff) const
  {
    ComputeCoefficients(z0,rho,coeff,false);
  }

  /**
   * Computes the central and remote coefficients for the conic section.  It is
   * taken from Dr. Ferenc Glueck's elcd3_2 program.
   */
  void KZHCoefficientGenerator<KConicSection>
  ::ComputeCoefficients(double z0,
			double rho_const,
			std::vector<double>& coeff,
			bool isCen) const
  {
    static int M=30;
    // slightly modified Newton-Cotes coefficients
    static double w9[10]={ 0.2803440531305107e0,0.1648702325837748e1,
			     -0.2027449845679092e0,0.2797927414021179e1,
			     -0.9761199294532843e0,0.2556499393738999e1,
			     0.1451083002645404e0,0.1311227127425048e1,
			     0.9324249063051143e0,0.1006631393298060e1};
    static double w[31];

    // Initialization of the integration weight factors (set once during the
    // calculation of the first source point)
    // if (!fIntegrationParameters)
    {
      // fIntegrationParameters = true;
      int m;
      for(m=0;m<=9;m++)
	w[m]=w9[m];
      for(m=10;m<=M-10;m++)
	w[m]=1.;
      for(m=M-9;m<=M;m++)
	w[m]=w9[M-m];
    }

    int n_coeffs = (int)coeff.size();
    std::vector<double> P1(n_coeffs,0);

    double Z = (fConicSection->GetZ0()+fConicSection->GetZ1())/2.-z0;
    double R = (fConicSection->GetR0()+fConicSection->GetR1())/2.;
    double L = sqrt((fConicSection->GetZ0()-fConicSection->GetZ1())*(fConicSection->GetZ0()-fConicSection->GetZ1())+(fConicSection->GetR0()-fConicSection->GetR1())*(fConicSection->GetR0()-fConicSection->GetR1()));

    double cos = (fConicSection->GetZ1()-fConicSection->GetZ0())/L;
    double sin = (fConicSection->GetR1()-fConicSection->GetR0())/L;

    double x = -Z*cos-R*sin;

    double sigma = Prefactor();
    double del = (double)L/M;
    double ro0,Rper,u0,rc,rcn,c;

    // integration loop (integrate over x)

    for (int m=0;m<=M;m++)
    {
      x = -L/2.+del*m;
      Z = (fConicSection->GetZ0()+fConicSection->GetZ1())/2. + x*(fConicSection->GetZ1()-fConicSection->GetZ0())/L;
      R = (fConicSection->GetR0()+fConicSection->GetR1())/2. + x*(fConicSection->GetR1()-fConicSection->GetR0())/L;
      ro0=sqrt(R*R+(Z-z0)*(Z-z0));
      if (ro0==0)
      {
	ro0=1.e-17;
      }
      Rper=R/ro0;

      u0=(Z-z0)/ro0;

      P1[0]=1.;
      P1[1]=u0;

      if (isCen)
	rc=rcn=rho_const/ro0;
      else
	rc=rcn=ro0/rho_const;

      c=sigma/(2.*KEMConstants::Eps0)*w[m]*del*Rper;

      if (isCen)
      {
	coeff[0]+=c*P1[0];
	coeff[1]+=c*rcn*P1[1];
      }
      else
      {
	coeff[0]+=c*rcn*P1[0];
	rcn*=rc;
	coeff[1]+=c*rcn*P1[1];      
      }

      // Legendre polynomial expansion loop
      for (int n=2;n<n_coeffs;n++)
      {
	rcn*=rc;
	P1[n]=KZHLegendreCoefficients::GetInstance()->Get(0,n)*u0*P1[n-1] -
	  KZHLegendreCoefficients::GetInstance()->Get(1,n)*P1[n-2];
	coeff[n]+=c*rcn*P1[n];
	if (rcn<1.e-17) break;
      }
    }
    return;
  }

  /**
   * Computes rho_cen/rem for the ring corresponding to a source point located
   * at z0.  
   */
  double KZHCoefficientGenerator<KConicSection>::ComputeRho(double z0, bool isCen) const
  {
    double Z = (fConicSection->GetZ0()+fConicSection->GetZ1())*.5 - z0;
    double R = (fConicSection->GetR0()+fConicSection->GetR1())*.5;

    double length = sqrt((fConicSection->GetZ0()-fConicSection->GetZ1())*
			   (fConicSection->GetZ0()-fConicSection->GetZ1()) +
			   (fConicSection->GetR0()-fConicSection->GetR1())*
			   (fConicSection->GetR0()-fConicSection->GetR1()));

    double cos = (fConicSection->GetZ1()-fConicSection->GetZ0())/length;
    double sin = (fConicSection->GetR1()-fConicSection->GetR0())/length;

    double x = -Z*cos-R*sin;
    double xm= length/2.;

    double rho_min;
    double rho_max;

    double rho_a = sqrt((Z-xm*cos)*(Z-xm*cos)+(R-xm*sin)*(R-xm*sin));
    double rho_b = sqrt((Z+xm*cos)*(Z+xm*cos)+(R+xm*sin)*(R+xm*sin));

    if (x>=xm)
    {
      rho_min = rho_b;
      rho_max = rho_a;
    }
    else if (x<=-xm)
    {
      rho_min = rho_a;
      rho_max = rho_b;
    }
    else
    {
      rho_min = sqrt((Z+x*cos)*(Z+x*cos)+(R+x*sin)*(R+x*sin));
      if (rho_a>rho_b)
	rho_max = rho_a;
      else
	rho_max = rho_b;
    }

    if (isCen)
      return rho_min;
    else
      return rho_max;
  }

  void KZHCoefficientGenerator<KConicSection>::GetExtrema(double& zMin,double& zMax) const
  {
    zMin = (fConicSection->GetZ0() < fConicSection->GetZ1() ?
	    fConicSection->GetZ0() : fConicSection->GetZ1());
    zMax = (fConicSection->GetZ0() > fConicSection->GetZ1() ?
	    fConicSection->GetZ0() : fConicSection->GetZ1());
  }
}
