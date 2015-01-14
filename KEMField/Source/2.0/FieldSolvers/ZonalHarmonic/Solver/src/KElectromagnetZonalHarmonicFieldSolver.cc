#include "KElectromagnetZonalHarmonicFieldSolver.hh"

#include <numeric>

namespace KEMField
{
  KEMThreeVector KZonalHarmonicFieldSolver<KMagnetostaticBasis>::VectorPotential(const KPosition& P) const
  {
    KEMThreeVector localP = fContainer.GetCoordinateSystem().ToLocal(P);

    KEMThreeVector A,B;

    if (UseCentralExpansion(localP))
      if (CentralExpansion(localP,A,B))
	return fContainer.GetCoordinateSystem().ToGlobal(A);

    if (UseRemoteExpansion(localP))
      if (RemoteExpansion(localP,A,B))
	return fContainer.GetCoordinateSystem().ToGlobal(A);

    if (fSubsetFieldSolvers.size()!=0)
    {
      VectorPotentialAccumulator accumulator(P);

      return std::accumulate(fSubsetFieldSolvers.begin(),
			     fSubsetFieldSolvers.end(),
			     A,
			     accumulator);
    }

    return fIntegratingFieldSolver.VectorPotential(P);
  }

  KEMThreeVector KZonalHarmonicFieldSolver<KMagnetostaticBasis>::MagneticField(const KPosition& P) const
  {
    KEMThreeVector localP = fContainer.GetCoordinateSystem().ToLocal(P);

    KEMThreeVector A,B;

    if (UseCentralExpansion(localP))
      if (CentralExpansion(localP,A,B))
	return fContainer.GetCoordinateSystem().ToGlobal(B);

    if (UseRemoteExpansion(localP))
      if (RemoteExpansion(localP,A,B))
       return fContainer.GetCoordinateSystem().ToGlobal(B);

    if (fSubsetFieldSolvers.size()!=0)
    {
      MagneticFieldAccumulator accumulator(P);
      return std::accumulate(fSubsetFieldSolvers.begin(),
                             fSubsetFieldSolvers.end(),
                             B,
                             accumulator);
    }

    return fIntegratingFieldSolver.MagneticField(P);
  }

  KGradient KZonalHarmonicFieldSolver<KMagnetostaticBasis>::MagneticFieldGradient(const KPosition& P) const
  {
    KEMThreeVector localP = fContainer.GetCoordinateSystem().ToLocal(P);

    KGradient g;

    if (UseCentralExpansion(localP))
      if (CentralGradientExpansion(localP,g))
	return fContainer.GetCoordinateSystem().ToGlobal(g);

    if (UseRemoteExpansion(localP))
      if (RemoteGradientExpansion(localP,g))
	return fContainer.GetCoordinateSystem().ToGlobal(g);

    if (fSubsetFieldSolvers.size()!=0)
    {
      MagneticFieldGradientAccumulator accumulator(P);

      return std::accumulate(fSubsetFieldSolvers.begin(),
			     fSubsetFieldSolvers.end(),
			     g,
			     accumulator);
    }

    return fIntegratingFieldSolver.MagneticFieldGradient(P);
  }

  bool KZonalHarmonicFieldSolver<KMagnetostaticBasis>::CentralExpansion(const KPosition& P,KEMThreeVector& vectorPotential,KEMThreeVector& magneticField) const
  {
    if (fContainer.GetCentralSourcePoints().empty())
    {
      vectorPotential[0] = vectorPotential[1] = vectorPotential[2] = 0.;
      magneticField[0] = magneticField[1] = magneticField[2] = 0.;
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
      vectorPotential[0] = vectorPotential[1] = vectorPotential[2] = magneticField[0] = magneticField[1] = 0.;
      magneticField[2] = sP.GetCoeff(0);
      return true;
    }

    // rho,u,s:
    double delz = z-sP.GetZ0();
    double rho   = sqrt(r*r+delz*delz);
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
    double A = -s*sP.GetRho()*sP.GetCoeff(0)*.5*rc;
    double Bz = sP.GetCoeff(0) + sP.GetCoeff(1)*rc*u;
    double Br = -s*sP.GetCoeff(1)*.5*rc;

    // flags for series convergence
    bool A_hasConverged  = false;
    bool Bz_hasConverged = false;
    bool Br_hasConverged = false;
    bool B_hasConverged  = false;

    // n-th A, Bz, Br terms in the series
    double Aplus,Bzplus,Brplus;         

    // (n-1)-th A, Bz, Br terms in the series (used for convergence)
    double lastAplus,lastBzplus,lastBrplus;
    lastAplus = lastBzplus = lastBrplus = 1.e30;

    // ratio of n-th A, Bz, Br terms to the series sums
    double A_ratio,Bz_ratio,Br_ratio;

    // sum of the last 4 B-field terms
    double B_delta[4] = {1.e20,1.e20,1.e20,1.e20};

    // Compute the series expansion
    for(unsigned int n=2;n<Ncoeffs-1;n++)
    {
      P1[n]=KZHLegendreCoefficients::GetInstance()->Get(0,n)*u*P1[n-1] -
	KZHLegendreCoefficients::GetInstance()->Get(1,n)*P1[n-2];
      P1p[n]=KZHLegendreCoefficients::GetInstance()->Get(2,n)*u*P1p[n-1] -
	KZHLegendreCoefficients::GetInstance()->Get(3,n)*P1p[n-2];

      // rcn = (rho/rho_cen)^n
      rcn*=rc;

      // n-th A, Bz, Br terms in the series
      Aplus=-sP.GetRho()*s*sP.GetCoeff(n-1)*(1.)/(n*(n+1.))*rcn*P1p[n];
      Bzplus=sP.GetCoeff(n)*rcn*P1[n];
      Brplus=-s*sP.GetCoeff(n)*(1.)/(n+1.)*rcn*P1p[n];

      A+=Aplus; Bz+=Bzplus; Br+=Brplus;

      // Conditions for series convergence:
      //   the last term in the series must be smaller than the current series
      //   sum by the given parameter, and smaller than the previous term
      A_ratio  = fContainer.GetParameters().GetConvergenceParameter()*fabs(A);
      Bz_ratio = fContainer.GetParameters().GetConvergenceParameter()*fabs(Bz);
      Br_ratio = fContainer.GetParameters().GetConvergenceParameter()*fabs(Br);


      // minimum of 8 iterations before convergence conditions are enforced
      if (n>8)
      {
	if((fabs(Aplus) < A_ratio && fabs(lastAplus) < A_ratio)
	   || r < fContainer.GetParameters().GetProximityToSourcePoint())
	  A_hasConverged = true;
	if(fabs(Bzplus) < Bz_ratio && fabs(lastBzplus) < Bz_ratio)
	  Bz_hasConverged =  true;
	if((fabs(Brplus) < Br_ratio && fabs(lastBrplus) < Br_ratio)
	   || r < fContainer.GetParameters().GetProximityToSourcePoint())
	  Br_hasConverged =  true;
      }

      B_delta[n%4] = fabs(Bzplus) + fabs(Brplus);

      if (B_delta[0]+B_delta[1]+B_delta[2]+B_delta[3]<(Bz_ratio+Br_ratio))
	B_hasConverged = true;

      if(A_hasConverged*Bz_hasConverged*Br_hasConverged*B_hasConverged == true)
	break;

      lastAplus=Aplus; lastBzplus=Bzplus; lastBrplus=Brplus;
    }

    if(A_hasConverged*Bz_hasConverged*Br_hasConverged*B_hasConverged == false)
    {
      return false;
    }

    magneticField[2] = Bz;
    vectorPotential[2] = 0;

    if (r<fContainer.GetParameters().GetProximityToSourcePoint())
      magneticField[0] = magneticField[1] = vectorPotential[0] = vectorPotential[1] = 0.;
    else
    {
      double cosine = P[0]/r;
      double sine = P[1]/r;
      magneticField[0] = cosine*Br;
      magneticField[1] = sine*Br;
      vectorPotential[0] = sine*A;
      vectorPotential[1] = -cosine*A;
    }

    return true;
  }

  bool KZonalHarmonicFieldSolver<KMagnetostaticBasis>::RemoteExpansion(const KPosition& P,KEMThreeVector& vectorPotential,KEMThreeVector& magneticField) const
  {
    if (fContainer.GetRemoteSourcePoints().empty())
    {
      vectorPotential[0] = vectorPotential[1] = vectorPotential[2] = 0.;
      magneticField[0] = magneticField[1] = magneticField[2] = 0.;
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

    // First 2 terms of the series:
    double rrn = rr*rr;
    double A = -sP.GetRho()*sP.GetCoeff(2)*s*.5*rrn;
    double Bz = 0;
    double Br = 0;

    // flags for series convergence
    bool A_hasConverged  = false;
    bool Bz_hasConverged = false;
    bool Br_hasConverged = false;
    bool B_hasConverged  = false;

    // n-th A, Bz, Br terms in the series
    double Aplus,Bzplus,Brplus;         

    // (n-1)-th A, Bz, Br terms in the series (used for convergence)
    double lastAplus,lastBzplus,lastBrplus;
    lastAplus = lastBzplus = lastBrplus = 1.e30;

    // ratio of n-th A, Bz, Br terms to the series sums
    double A_ratio,Bz_ratio,Br_ratio;

    // sum of the last 4 B-field terms
    double B_delta[4] = {1.e20,1.e20,1.e20,1.e20};

    // Compute the series expansion
    for(unsigned int n=2;n<Ncoeffs-1;n++)
    {
      P1[n]=KZHLegendreCoefficients::GetInstance()->Get(0,n)*u*P1[n-1] -
	KZHLegendreCoefficients::GetInstance()->Get(1,n)*P1[n-2];
      P1p[n]=KZHLegendreCoefficients::GetInstance()->Get(2,n)*u*P1p[n-1] -
	KZHLegendreCoefficients::GetInstance()->Get(3,n)*P1p[n-2];

      // rrn = (rho_rem/rho)^(n+1)
      rrn*=rr;

      // n-th A, Bz, Br terms in the series
      Aplus=-sP.GetRho()*sP.GetCoeff(n+1)*s*(1.)/(n*(n+1.))*rrn*P1p[n];
      Bzplus=sP.GetCoeff(n)*rrn*P1[n];
      Brplus=sP.GetCoeff(n)*s/n*rrn*P1p[n];

      A+=Aplus; Bz+=Bzplus; Br+=Brplus;

      // Conditions for series convergence:
      //   the last term in the series must be smaller than the current series
      //   sum by the given parameter, and smaller than the previous term
      A_ratio  = fContainer.GetParameters().GetConvergenceParameter()*fabs(A);
      Bz_ratio = fContainer.GetParameters().GetConvergenceParameter()*fabs(Bz);
      Br_ratio = fContainer.GetParameters().GetConvergenceParameter()*fabs(Br);

      // minimum of 8 iterations before convergence conditions are enforced
      //     if (n>8)
      //     {
      if((fabs(Aplus) < A_ratio && fabs(lastAplus) < A_ratio)
	 || r < fContainer.GetParameters().GetProximityToSourcePoint())
	A_hasConverged = true;
      if(fabs(Bzplus) < Bz_ratio   && fabs(lastBzplus) < Bz_ratio)
	Bz_hasConverged =  true;
      if((fabs(Brplus) < Br_ratio  && fabs(lastBrplus) < Br_ratio)
	 || r < fContainer.GetParameters().GetProximityToSourcePoint())
	Br_hasConverged =  true;
      //     }

      B_delta[n%4] = fabs(Bzplus) + fabs(Brplus);

      if (B_delta[0]+B_delta[1]+B_delta[2]+B_delta[3]<(Bz_ratio+Br_ratio))
	B_hasConverged = true;

      if(A_hasConverged*Bz_hasConverged*Br_hasConverged*B_hasConverged == true)
	break;

      lastAplus=Aplus; lastBzplus=Bzplus; lastBrplus=Brplus;
    }

    if (A_hasConverged*Bz_hasConverged*Br_hasConverged*B_hasConverged == false)
      return false;

    magneticField[2] = Bz;
    vectorPotential[2] = 0;

    if (r<fContainer.GetParameters().GetProximityToSourcePoint())
      magneticField[0] = magneticField[1] = vectorPotential[0] = vectorPotential[1] = 0.;
    else
    {
      double cosine = P[0]/r;
      double sine = P[1]/r;
      magneticField[0] = cosine*Br;
      magneticField[1] = sine*Br;
      vectorPotential[0] = sine*A;
      vectorPotential[1] = -cosine*A;
    }

    return true;
  }

  bool KZonalHarmonicFieldSolver<KMagnetostaticBasis>::CentralGradientExpansion(const KPosition& P,KGradient& g) const
  {
    if (fContainer.GetCentralSourcePoints().empty())
    {
      g[0] = g[1] = g[2] = g[3] = g[4] = g[5] = g[6] = g[7] = g[8] = 0.;
      return true;
    }

    double r = sqrt(P[0]*P[0]+P[1]*P[1]);
    double z = P[2];

    const KZonalHarmonicSourcePoint& sP =
      *(fContainer.GetCentralSourcePoints().at(fmCentralSPIndex));

    if (r<fContainer.GetParameters().GetProximityToSourcePoint())
      return false;

    // rho,u,s:
    double delz = z-sP.GetZ0();
    double rho   = sqrt(r*r+delz*delz);
    double u    = delz/rho;
    double u2   = u*u;
    double s    = r/rho;
    double s2   = s*s;

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
    P1[2]=KZHLegendreCoefficients::GetInstance()->Get(0,2)*u*P1[1] -
      KZHLegendreCoefficients::GetInstance()->Get(1,2)*P1[0];
    P1p[0]=0.; P1p[1]=1.;
    P1p[2]=KZHLegendreCoefficients::GetInstance()->Get(2,2)*u*P1p[1] -
      KZHLegendreCoefficients::GetInstance()->Get(3,2)*P1p[0];

    // First 2 terms of the series:
    double rcn = rc;
    double A = -s*sP.GetRho()*sP.GetCoeff(0)*.5*rc;
    double Bz = sP.GetCoeff(0) + sP.GetCoeff(1)*rc*u;
    double Br = -s*sP.GetCoeff(1)*.5*rc;
    double Bzz = sP.GetCoeff(1)/sP.GetRho();
    double Bzr = 0.;
    double Brr=-sP.GetCoeff(1)*.5/(sP.GetRho()*s2)*
      ((s2-2.*u2) + (s2+4.*u2)*u2 - 2.*u2*P1[2]);

    // flags for series convergence
    bool A_hasConverged  = false;
    bool Bz_hasConverged = false;
    bool Br_hasConverged = false;
    bool B_hasConverged  = false;
    bool Brr_hasConverged = false;
    bool Bzr_hasConverged = false;
    bool Bzz_hasConverged = false;

    // n-th A, Bz, Br terms in the series
    double Aplus,Bzplus,Brplus,Bzzplus,Bzrplus,Brrplus;   

    // (n-1)-th A, Bz, Br terms in the series (used for convergence)
    double lastAplus,lastBzplus,lastBrplus;
    lastAplus = lastBzplus = lastBrplus = 1.e30;

    double lastBrrplus,lastBrzplus,lastBzrplus,lastBzzplus;
    lastBrrplus = lastBrzplus = lastBzrplus = lastBzzplus = 1.e30;

    // ratio of n-th A, Bz, Br terms to the series sums
    double A_ratio,Bz_ratio,Br_ratio;
    double Brr_ratio,Bzr_ratio,Bzz_ratio;

    // sum of the last 4 B-field terms
    double B_delta[4] = {1.e20,1.e20,1.e20,1.e20};

    // Compute the series expansion
    for(unsigned int n=2;n<Ncoeffs-2;n++)
    {
      P1[n+1]=KZHLegendreCoefficients::GetInstance()->Get(0,n+1)*u*P1[n] -
	KZHLegendreCoefficients::GetInstance()->Get(1,n+1)*P1[n-1];
      P1p[n+1]=KZHLegendreCoefficients::GetInstance()->Get(2,n+1)*u*P1p[n] -
	KZHLegendreCoefficients::GetInstance()->Get(3,n+1)*P1p[n-1];

      // rcn = (rho/rho_cen)^n
      rcn*=rc;

      // n-th A, Bz, Br terms in the series
      Aplus=-sP.GetRho()*s*sP.GetCoeff(n-1)*(1.)/(n*(n+1.))*rcn*P1p[n];
      Bzplus=sP.GetCoeff(n)*rcn*P1[n];
      Brplus=-s*sP.GetCoeff(n)*(1.)/(n+1.)*rcn*P1p[n];
      Bzzplus=sP.GetCoeff(n)*rcn*n/rho*P1[n-1];

      Bzrplus=-s*sP.GetCoeff(n)*rcn/rho*P1p[n-1];

      Brrplus=-sP.GetCoeff(n)*(n/(n+1.))*rcn/(rho*s2)*
      	((n*s2-(n+1.)*u2)*P1[n-1] +
      	 u*(s2+2.*(n+1.)*u2)*P1[n] -
      	 (n+1.)*u2*P1[n+1]);

      A+=Aplus; Bz+=Bzplus; Br+=Brplus;
      Bzz+=Bzzplus; Bzr+=Bzrplus; Brr+=Brrplus;

      // Conditions for series convergence:
      //   the last term in the series must be smaller than the current series
      //   sum by the given parameter, and smaller than the previous term
      A_ratio  = fContainer.GetParameters().GetConvergenceParameter()*fabs(A);
      Bz_ratio = fContainer.GetParameters().GetConvergenceParameter()*fabs(Bz);
      Br_ratio = fContainer.GetParameters().GetConvergenceParameter()*fabs(Br);
      Brr_ratio= fContainer.GetParameters().GetConvergenceParameter()*fabs(Brr);
      Bzr_ratio= fContainer.GetParameters().GetConvergenceParameter()*fabs(Bzr);
      Bzz_ratio= fContainer.GetParameters().GetConvergenceParameter()*fabs(Bzz);


      // minimum of 8 iterations before convergence conditions are enforced
      if (n>8)
      {
	if((fabs(Aplus) < A_ratio && fabs(lastAplus) < A_ratio)
	   || r < fContainer.GetParameters().GetProximityToSourcePoint())
	  A_hasConverged = true;
	if(fabs(Bzplus) < Bz_ratio && fabs(lastBzplus) < Bz_ratio)
	  Bz_hasConverged =  true;
	if((fabs(Brplus) < Br_ratio && fabs(lastBrplus) < Br_ratio)
	   || r < fContainer.GetParameters().GetProximityToSourcePoint())
	  Br_hasConverged =  true;
	if((fabs(Brrplus) < Brr_ratio && fabs(lastBrrplus) < Brr_ratio)
	   || r < fContainer.GetParameters().GetProximityToSourcePoint())
	  Brr_hasConverged =  true;
	if((fabs(Bzrplus) < Bzr_ratio && fabs(lastBzrplus) < Bzr_ratio)
	   || r < fContainer.GetParameters().GetProximityToSourcePoint())
	  Bzr_hasConverged =  true;
	if((fabs(Bzzplus) < Bzz_ratio && fabs(lastBzzplus) < Bzz_ratio)
	   || r < fContainer.GetParameters().GetProximityToSourcePoint())
	  Bzz_hasConverged =  true;
      }

      B_delta[n%4] = fabs(Bzplus) + fabs(Brplus);

      if (B_delta[0]+B_delta[1]+B_delta[2]+B_delta[3]<(Bz_ratio+Br_ratio))
	B_hasConverged = true;

      if(A_hasConverged*Bz_hasConverged*Br_hasConverged*B_hasConverged*Brr_hasConverged*Bzr_hasConverged*Bzz_hasConverged == true)
	break;

      lastAplus=Aplus; lastBzplus=Bzplus; lastBrplus=Brplus;
      lastBrrplus=Brrplus; lastBzrplus=Bzrplus; lastBzzplus=Bzzplus;
    }

    if(A_hasConverged*Bz_hasConverged*Br_hasConverged*B_hasConverged*Brr_hasConverged*Bzr_hasConverged*Bzz_hasConverged == false)
    {
      return false;
    }

    double cosine = P[0]/r;
    double sine = P[1]/r;
    g[0]=Brr*cosine*cosine + 1./r*Br*sine*sine;
    g[1]=g[3]=Brr*cosine*sine - 1./r*Br*cosine*sine;
    g[2]=Bzr*cosine;
    g[4]=Brr*sine*sine + 1./r*Br*cosine*cosine;
    g[5]=Bzr*sine;
    g[6]=Bzr*cosine;
    g[7]=Bzr*sine;
    g[8] = Bzz;

    return true;
  }

  bool KZonalHarmonicFieldSolver<KMagnetostaticBasis>::RemoteGradientExpansion(const KPosition& P,KGradient& g) const
  {
    if (fContainer.GetRemoteSourcePoints().empty())
    {
      g[0] = g[1] = g[2] = g[3] = g[4] = g[5] = g[6] = g[7] = g[8] = 0.;
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
    double s2   = s*s;

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
  P1[2]=KZHLegendreCoefficients::GetInstance()->Get(0,2)*u*P1[1] -
    KZHLegendreCoefficients::GetInstance()->Get(1,2)*P1[0];
    P1p[0]=0.; P1p[1]=1.;
  P1p[2]=KZHLegendreCoefficients::GetInstance()->Get(2,2)*u*P1p[1] -
    KZHLegendreCoefficients::GetInstance()->Get(3,2)*P1p[0];

    // First 2 terms of the series:
    double rrn = rr*rr;
    double A = -sP.GetRho()*sP.GetCoeff(2)*s*.5*rrn;
    double Bz = 0;
    double Br = 0;
    double Bzz = 0;
    double Bzr = 0;
    double Brr = 0;

    // flags for series convergence
    bool A_hasConverged  = false;
    bool Bz_hasConverged = false;
    bool Br_hasConverged = false;
    bool B_hasConverged  = false;
    bool Brr_hasConverged = false;
    bool Bzr_hasConverged = false;
    bool Bzz_hasConverged = false;

    // n-th A, Bz, Br terms in the series
    double Aplus,Bzplus,Brplus,Bzzplus,Bzrplus,Brrplus;

    // (n-1)-th A, Bz, Br terms in the series (used for convergence)
    double lastAplus,lastBzplus,lastBrplus;
    lastAplus = lastBzplus = lastBrplus = 1.e30;

    double lastBrrplus,lastBzrplus,lastBzzplus;
    lastBrrplus = lastBzrplus = lastBzzplus = 1.e30;

    // ratio of n-th A, Bz, Br terms to the series sums
    double A_ratio,Bz_ratio,Br_ratio;
    double Brr_ratio,Bzr_ratio,Bzz_ratio;

    // sum of the last 4 B-field terms
    double B_delta[4] = {1.e20,1.e20,1.e20,1.e20};

    // Compute the series expansion
    for(unsigned int n=2;n<Ncoeffs-2;n++)
    {
      P1[n+1]=KZHLegendreCoefficients::GetInstance()->Get(0,n+1)*u*P1[n] -
	KZHLegendreCoefficients::GetInstance()->Get(1,n+1)*P1[n-1];
      P1p[n+1]=KZHLegendreCoefficients::GetInstance()->Get(2,n+1)*u*P1p[n] -
	KZHLegendreCoefficients::GetInstance()->Get(3,n+1)*P1p[n-1];

      // rrn = (rho_rem/rho)^(n+1)
      rrn*=rr;

      // n-th A, Bz, Br terms in the series
      Aplus=-sP.GetRho()*sP.GetCoeff(n+1)*s*(1.)/(n*(n+1.))*rrn*P1p[n];
      Bzplus=sP.GetCoeff(n)*rrn*P1[n];
      Brplus=sP.GetCoeff(n)*s/n*rrn*P1p[n];
      Bzzplus=-sP.GetCoeff(n)*(n+1.)*rrn/rho*P1[n+1];
      Bzrplus=-sP.GetCoeff(n)*rrn*s/rho*P1p[n+1];
      Brrplus=sP.GetCoeff(n)*rrn/rho*
	((n+1)*u*P1[n]-(s2+1./n)*P1p[n]);

      A+=Aplus; Bz+=Bzplus; Br+=Brplus;
      Bzz+=Bzzplus; Bzr+=Bzrplus; Brr+=Brrplus;

      // Conditions for series convergence:
      //   the last term in the series must be smaller than the current series
      //   sum by the given parameter, and smaller than the previous term
      A_ratio  = fContainer.GetParameters().GetConvergenceParameter()*fabs(A);
      Bz_ratio = fContainer.GetParameters().GetConvergenceParameter()*fabs(Bz);
      Br_ratio = fContainer.GetParameters().GetConvergenceParameter()*fabs(Br);
      Brr_ratio= fContainer.GetParameters().GetConvergenceParameter()*fabs(Brr);
      Bzr_ratio= fContainer.GetParameters().GetConvergenceParameter()*fabs(Bzr);
      Bzz_ratio= fContainer.GetParameters().GetConvergenceParameter()*fabs(Bzz);

      // minimum of 8 iterations before convergence conditions are enforced
      //     if (n>8)
      //     {
      if((fabs(Aplus) < A_ratio && fabs(lastAplus) < A_ratio)
	 || r < fContainer.GetParameters().GetProximityToSourcePoint())
	A_hasConverged = true;
      if(fabs(Bzplus) < Bz_ratio   && fabs(lastBzplus) < Bz_ratio)
	Bz_hasConverged =  true;
      if((fabs(Brplus) < Br_ratio  && fabs(lastBrplus) < Br_ratio)
	 || r < fContainer.GetParameters().GetProximityToSourcePoint())
	Br_hasConverged =  true;
	if((fabs(Brrplus) < Brr_ratio && fabs(lastBrrplus) < Brr_ratio)
	   || r < fContainer.GetParameters().GetProximityToSourcePoint())
	  Brr_hasConverged =  true;
	if((fabs(Bzrplus) < Bzr_ratio && fabs(lastBzrplus) < Bzr_ratio)
	   || r < fContainer.GetParameters().GetProximityToSourcePoint())
	  Bzr_hasConverged =  true;
	if((fabs(Bzzplus) < Bzz_ratio && fabs(lastBzzplus) < Bzz_ratio)
	   || r < fContainer.GetParameters().GetProximityToSourcePoint())
	  Bzz_hasConverged =  true;
      //     }

      B_delta[n%4] = fabs(Bzplus) + fabs(Brplus);

      if (B_delta[0]+B_delta[1]+B_delta[2]+B_delta[3]<(Bz_ratio+Br_ratio))
	B_hasConverged = true;

      if(A_hasConverged*Bz_hasConverged*Br_hasConverged*B_hasConverged*Brr_hasConverged*Bzr_hasConverged*Bzz_hasConverged == true)
	break;

      lastAplus=Aplus; lastBzplus=Bzplus; lastBrplus=Brplus;
      lastBrrplus=Brrplus; lastBzrplus=Bzrplus; lastBzzplus=Bzzplus;
    }

    if(A_hasConverged*Bz_hasConverged*Br_hasConverged*B_hasConverged*Brr_hasConverged*Bzr_hasConverged*Bzz_hasConverged == false)
    {
      return false;
    }

    g[8] = Bzz;

    if (r<fContainer.GetParameters().GetProximityToSourcePoint())
      g[0] = g[1] = g[2] = g[3] = g[4] = g[5] = g[6] = g[7] = 0.;
    else
    {
      double cosine = P[0]/r;
      double sine = P[1]/r;
      g[0]=Brr*cosine*cosine + 1./r*Br*sine*sine;
      g[1]=g[3]=Brr*cosine*sine - 1./r*Br*cosine*sine;
      g[2]=Bzr*cosine;
      g[4]=Brr*sine*sine + 1./r*Br*cosine*cosine;
      g[5]=Bzr*sine;
      g[6]=Bzr*cosine;
      g[7]=Bzr*sine;
    }

    return true;
  }
}
