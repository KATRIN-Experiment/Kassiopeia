#include "KElectrostaticZonalHarmonicFieldSolver.hh"

#include <numeric>

// #include "KShanksTransformation.hh"

namespace KEMField
{
  double KZonalHarmonicFieldSolver<KElectrostaticBasis>::Potential(const KPosition& P) const
  {
    double phi = 0;
    KEMThreeVector E;

    if (UseCentralExpansion(P))
      if (CentralExpansion(P,phi,E))
	return phi;

    if (UseRemoteExpansion(P))
      if (RemoteExpansion(P,phi,E))
	return phi;

    if (fSubsetFieldSolvers.size()!=0)
    {
      PotentialAccumulator accumulator(P);

      return std::accumulate(fSubsetFieldSolvers.begin(),
			     fSubsetFieldSolvers.end(),
			     phi,
			     accumulator);
    }

    return fIntegratingFieldSolver.Potential(P);
  }

  KEMThreeVector KZonalHarmonicFieldSolver<KElectrostaticBasis>::ElectricField(const KPosition& P) const
  {
    double phi = 0;
    KEMThreeVector E;

    if (UseCentralExpansion(P))
      if (CentralExpansion(P,phi,E))
	return E;

    if (UseRemoteExpansion(P))
      if (RemoteExpansion(P,phi,E))
	return E;

    if (fSubsetFieldSolvers.size()!=0)
    {
      ElectricFieldAccumulator accumulator(P);

      return std::accumulate(fSubsetFieldSolvers.begin(),
			     fSubsetFieldSolvers.end(),
			     E,
			     accumulator);
    }

    return fIntegratingFieldSolver.ElectricField(P);
  }

  bool KZonalHarmonicFieldSolver<KElectrostaticBasis>::CentralExpansion(const KPosition& P,double& potential,KEMThreeVector& electricField) const
  {
    if (fContainer.GetCentralSourcePoints().empty())
    {
      potential = 0.;
      electricField[0] = electricField[1] = electricField[2] = 0.;
      return true;
    }

    double r = sqrt(P[0]*P[0]+P[1]*P[1]);
    double z = P[2];

    const KZonalHarmonicSourcePoint& sP =
      *(fContainer.GetCentralSourcePoints().at(fmCentralSPIndex));

    // if the field point is very close to the source point
    if (r<fContainer.GetParameters().GetProximityToSourcePoint() &&
	fabs(z-sP.GetZ0())<fContainer.GetParameters().GetProximityToSourcePoint())
    {
      potential = sP.GetCoeff(0);
      electricField[2] = -sP.GetCoeff(1)/sP.GetRho();
      electricField[0] = electricField[1] = 0;
      return true;
    }

    // ro,u,s:
    double delz = z-sP.GetZ0();
    double rho  = sqrt(r*r+delz*delz);
    double u    = delz/rho;
    double s    = r/rho;

    // Convergence ratio:
    double rc = rho/sP.GetRho();

    // Create the Legendre polynomial arrays
    unsigned int Ncoeffs = sP.GetNCoeffs();
    static std::vector<double> P1(Ncoeffs);
    static std::vector<double> P1p(Ncoeffs);
    if (Ncoeffs > P1.size())
    {
      P1.resize(Ncoeffs,0.);
      P1p.resize(Ncoeffs,0.);
    }

    P1[0]=1.; P1[1]=u;
    P1p[0]=0.; P1p[1]=1.;

    // First 2 terms of the series:
    double rcn = rc;
    double Phi = sP.GetCoeff(0) + sP.GetCoeff(1)*rc*u;
    double Ez = (sP.GetCoeff(1) + sP.GetCoeff(2)*2.*rc*u);
    double Er = sP.GetCoeff(2)*rc;

    // flags for series convergence
    bool Phi_hasConverged = false;
    bool Ez_hasConverged  = false;
    bool Er_hasConverged  = false;

    // n-th Phi, Ez, Er terms in the series
    double Phiplus,Ezplus,Erplus;

    // (n-1)-th Phi, Ez, Er terms in the series (used for convergence)
    double lastPhiplus,lastEzplus,lastErplus;
    lastPhiplus = lastEzplus = lastErplus = 1.e30;

    // ratio of n-th Phi, Ez, Er terms to the series sums
    double Phi_ratio,Ez_ratio,Er_ratio;

    // Compute the series expansion
    for (unsigned int n=2;n<Ncoeffs-1;n++)
    {
      P1[n]=KZHLegendreCoefficients::GetInstance()->Get(0,n)*u*P1[n-1] -
	KZHLegendreCoefficients::GetInstance()->Get(1,n)*P1[n-2];
      P1p[n]=KZHLegendreCoefficients::GetInstance()->Get(2,n)*u*P1p[n-1] -
	KZHLegendreCoefficients::GetInstance()->Get(3,n)*P1p[n-2];

      // rcn = (rho/rho_cen)^n
      rcn*=rc;

      // n-th Phi, Ez, Er terms in the series
      Phiplus=sP.GetCoeff(n)*rcn*P1[n];
      Ezplus=sP.GetCoeff(n+1)*(n+1.)*rcn*P1[n];
      Erplus=sP.GetCoeff(n+1)*rcn*P1p[n];

      Phi+=Phiplus; Ez+=Ezplus; Er+=Erplus;

      // Conditions for series convergence:
      //   the last term in the series must be smaller than the current series
      //   sum by the given parameter, and smaller than the previous term
      Phi_ratio = fContainer.GetParameters().GetConvergenceParameter()*fabs(Phi);
      Ez_ratio  = fContainer.GetParameters().GetConvergenceParameter()*fabs(Ez);
      Er_ratio  = fContainer.GetParameters().GetConvergenceParameter()*fabs(Er);

      if (fabs(Phiplus) < Phi_ratio && fabs(lastPhiplus) < Phi_ratio)
	Phi_hasConverged = true;
      if (fabs(Ezplus) < Ez_ratio   && fabs(lastEzplus) < Ez_ratio)
	Ez_hasConverged =  true;
      if ((fabs(Erplus) < Er_ratio  && fabs(lastErplus) < Er_ratio)
	 || r < fContainer.GetParameters().GetProximityToSourcePoint())
	Er_hasConverged =  true;

      if (Phi_hasConverged*Ez_hasConverged*Er_hasConverged == true) break;

      lastPhiplus=Phiplus; lastEzplus=Ezplus; lastErplus=Erplus;
    }

    if (Phi_hasConverged*Ez_hasConverged*Er_hasConverged == false)
      return false;

    Ez*=-1./sP.GetRho();
    Er*=s/sP.GetRho();

    potential = Phi;

    electricField[2] = Ez;

    if (r<fContainer.GetParameters().GetProximityToSourcePoint())
      electricField[0] = electricField[1] = 0.;
    else
    {
      double cosine = P[0]/r;
      double sine = P[1]/r;
      electricField[0] = cosine*Er;
      electricField[1] = sine*Er;
    }

    return true;
  }

  bool KZonalHarmonicFieldSolver<KElectrostaticBasis>::RemoteExpansion(const KPosition& P,double& potential,KEMThreeVector& electricField) const
  {
    if (fContainer.GetRemoteSourcePoints().empty())
    {
      potential = 0.;
      electricField[0] = electricField[1] = electricField[2] = 0.;
      return true;
    }

    double r = sqrt(P[0]*P[0]+P[1]*P[1]);
    double z = P[2];

    const KZonalHarmonicSourcePoint& sP =
      *(fContainer.GetRemoteSourcePoints().at(fmRemoteSPIndex));

    // rho,u,s:
    double delz = z-sP.GetZ0();
    double rho  = sqrt(r*r+delz*delz);
    if (rho<1.e-9) rho=1.e-9;
    double u    = delz/rho;
    double s    = r/rho;

    // Convergence ratio:
    double rr = sP.GetRho()/rho;  // convergence ratio

    // Create the Legendre polynomial arrays
    unsigned int Ncoeffs = sP.GetNCoeffs();
    static std::vector<double> P1(Ncoeffs);
    static std::vector<double> P1p(Ncoeffs);
    if (Ncoeffs > P1.size())
    {
      P1.resize(Ncoeffs,0.);
      P1p.resize(Ncoeffs,0.);
    }

    P1[0]=1.; P1[1]=u;
    P1p[0]=0.; P1p[1]=1.;

    // series loop starts at n = 2, so we manually compute the first three terms

    // (n-1)-th Phi, Ez, Er terms in the series (used for convergence)
    double Phi_n_1 = sP.GetCoeff(0)*rr;
    double Ez_n_1 = 0.;
    double Er_n_1 = 0.;

    // n-th Phi, Ez, Er terms in the series
    double rrn = rr*rr;
    double Phi_n = sP.GetCoeff(1)*rrn*u;
    double Ez_n = sP.GetCoeff(0)*rrn*u;
    double Er_n = sP.GetCoeff(0)*rrn;

    // First 3 terms of the series:
    double Phi = Phi_n_1 + Phi_n;
    double Ez = Ez_n_1 + Ez_n;
    double Er = Er_n_1 + Er_n;

    // static KShanksTransformation shanksTransform;
    // double Phi_S[4] = {0.,0.,0.,0.};
    // double Ez_S[4]  = {0.,0.,0.,0.};
    // double Er_S[4]  = {0.,0.,0.,0.};

    // flags for series convergence
    bool Phi_hasConverged = false;
    bool Ez_hasConverged  = false;
    bool Er_hasConverged  = false;

    // ratio of n-th Phi, Ez, Er terms to the series sums
    double Phi_ratio,Ez_ratio,Er_ratio;

    // Compute the series expansion
    for (unsigned int n=2;n<Ncoeffs;n++)
    {
      P1[n]=KZHLegendreCoefficients::GetInstance()->Get(0,n)*u*P1[n-1] -
	KZHLegendreCoefficients::GetInstance()->Get(1,n)*P1[n-2];
      P1p[n]=KZHLegendreCoefficients::GetInstance()->Get(2,n)*u*P1p[n-1] -
	KZHLegendreCoefficients::GetInstance()->Get(3,n)*P1p[n-2];

      // rrn = (rho_rem/rho)^(n+1)
      rrn*=rr;

      // n-th Phi, Ez, Er terms in the series
      Phi_n=sP.GetCoeff(n)*rrn*P1[n];
      Ez_n=sP.GetCoeff(n-1)*n*rrn*P1[n];
      Er_n=sP.GetCoeff(n-1)*rrn*P1p[n];

      Phi+=Phi_n; Ez+=Ez_n; Er+=Er_n;

      // Phi_S[3] = Phi_S[2];
      // Phi_S[2] = Phi_S[1];
      // Phi_S[1] = Phi;

      // Ez_S[3] = Ez_S[2];
      // Ez_S[2] = Ez_S[1];
      // Ez_S[1] = Ez;

      // Er_S[3] = Er_S[2];
      // Er_S[2] = Er_S[1];
      // Er_S[1] = Er;

      // if (n > 5)
      // {
      // 	Phi_S[0] = shanksTransform(Phi_S[3],Phi_S[2],Phi_S[1]);
      // 	Ez_S[0]  = shanksTransform(Ez_S[3],Ez_S[2],Ez_S[1]);
      // 	Er_S[0]  = shanksTransform(Er_S[3],Er_S[2],Er_S[1]);
      // }

      // Conditions for series convergence:
      //   the last term in the series must be smaller than the current series
      //   sum by the given parameter, and smaller than the previous term
      Phi_ratio = fContainer.GetParameters().GetConvergenceParameter()*fabs(Phi);
      Ez_ratio  = fContainer.GetParameters().GetConvergenceParameter()*fabs(Ez);
      Er_ratio  = fContainer.GetParameters().GetConvergenceParameter()*fabs(Er);

      if (fabs(Phi_n) < Phi_ratio && fabs(Phi_n_1) < Phi_ratio)
	Phi_hasConverged = true;
      if (fabs(Ez_n) < Ez_ratio   && fabs(Ez_n_1) < Ez_ratio)
	Ez_hasConverged =  true;
      if ((fabs(Er_n) < Er_ratio  && fabs(Er_n_1) < Er_ratio)
	 || r < fContainer.GetParameters().GetProximityToSourcePoint())
	Er_hasConverged =  true;

      if (Phi_hasConverged*Ez_hasConverged*Er_hasConverged == true) break;

      Phi_n_1=Phi_n; Ez_n_1=Ez_n; Er_n_1=Er_n;
    }

    if (Phi_hasConverged*Ez_hasConverged*Er_hasConverged == false)
      return false;

    // Phi = Phi_S[0];
    // Er = Er_S[0];
    // Ez = Ez_S[0];

    Ez*=1./sP.GetRho();
    Er*=s/sP.GetRho();

    potential = Phi;

    electricField[2] = Ez;

    if (r<fContainer.GetParameters().GetProximityToSourcePoint())
      electricField[0] = electricField[1]=0;
    else
    {
      double cosine = P[0]/r;
      double sine = P[1]/r;
      electricField[0] = cosine*Er;
      electricField[1] = sine*Er;
    }

    return true;
  }
}
