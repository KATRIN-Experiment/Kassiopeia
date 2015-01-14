#ifndef KZONALHARMONICCOMPUTER_DEF
#define KZONALHARMONICCOMPUTER_DEF

#include "KZonalHarmonicContainer.hh"

namespace KEMField
{
  template <class Basis>
  class KZonalHarmonicFieldSolver;

  template <class Basis>
  class KZonalHarmonicComputer
  {
  public:
    typedef KZonalHarmonicTrait<Basis> ZonalHarmonicType;
    typedef typename ZonalHarmonicType::Integrator Integrator;
    typedef KZonalHarmonicContainer<Basis> Container;
    typedef typename ZonalHarmonicType::Container ElementContainer;
    typedef std::vector<KZonalHarmonicSourcePoint*> SourcePointVector;
    typedef std::vector<KZonalHarmonicFieldSolver<Basis>*> FieldSolverVector;

    void Initialize();

  protected:
    KZonalHarmonicComputer(Container& container,Integrator& integrator) :
      fContainer(container),
      fIntegrator(integrator),
      fmCentralSPIndex(-1),
      fmRemoteSPIndex(-1) {}

    virtual ~KZonalHarmonicComputer() {}
  public:
    bool UseCentralExpansion(const KPosition& P) const;
    bool UseRemoteExpansion(const KPosition& P) const;
  protected:
    Container& fContainer;
    Integrator& fIntegrator;
    FieldSolverVector fSubsetFieldSolvers;

    mutable int fmCentralSPIndex;
    mutable int fmRemoteSPIndex;
  };

  template <class Basis>
  bool KZonalHarmonicComputer<Basis>::UseCentralExpansion(const KPosition& P) const
  {
    if (fContainer.GetCentralSourcePoints().empty()) return false;

    double rcmin = 1.e20;

    double r = sqrt(P[0]*P[0]+P[1]*P[1]);
    double z = P[2];

    double delz;
    double rho;
    double rc = 0;

    // Check neighboring SP's if this is not the function's first call...
    if (fmCentralSPIndex != -1)
    {
      for (int i=fmCentralSPIndex-2;i<=fmCentralSPIndex+2;i++)
      {
	if (i<0 || i>=(int)fContainer.GetCentralSourcePoints().size())
	  continue;

	const KZonalHarmonicSourcePoint& sP = *(fContainer.GetCentralSourcePoints().at(i));

	delz = z-sP.GetZ0();
	rho  = sqrt(r*r+delz*delz);
	rc=rho/sP.GetRho();

	if (rc<rcmin)
	{
	  rcmin=rc;
	  fmCentralSPIndex = i;
	}
      }
    }

    // ...if this is the function's first call, OR if Legendre
    // polynomial expansion does not converge, check the whole list

    if (fmCentralSPIndex == -1 || rc>fContainer.GetParameters().GetConvergenceRatio())
    {
      rcmin = 1.e20;

      for (unsigned int i=0;i<fContainer.GetCentralSourcePoints().size();i++)
      {
	const KZonalHarmonicSourcePoint& sP = *(fContainer.GetCentralSourcePoints().at(i));

	delz = z-sP.GetZ0();
	rho  = sqrt(r*r+delz*delz);
	rc=rho/sP.GetRho();

	if (rc<rcmin)
	{
	  rcmin=rc;
	  fmCentralSPIndex = i;
	}
      }
    }

    if (rcmin>fContainer.GetParameters().GetConvergenceRatio())
      return false;
    else
      return true;
  }

  template <class Basis>
  bool KZonalHarmonicComputer<Basis>::UseRemoteExpansion(const KPosition& P) const
  {
    if (fContainer.GetRemoteSourcePoints().empty()) return false;

    double rrmin = 1.e20;

    double r = sqrt(P[0]*P[0]+P[1]*P[1]);
    double z = P[2];

    double delz;
    double rho;
    double rr = 0;

    // Check neighboring SP's if this is not the function's first call...
    if (fmRemoteSPIndex!=-1)
    {
      for (int i=fmRemoteSPIndex-1;i<=fmRemoteSPIndex+1;i++)
      {
	if (i<0 || i>=(int)fContainer.GetRemoteSourcePoints().size())
	  continue;

	const KZonalHarmonicSourcePoint& sP = *(fContainer.GetRemoteSourcePoints().at(i));

	delz = z-sP.GetZ0();
	rho  = sqrt(r*r+delz*delz);
	rr=sP.GetRho()/rho;

	if (rr<rrmin)
	{
	  rrmin=rr;
	  fmRemoteSPIndex = i;
	}
      }
    }


    // ...if this is the function's first call, OR if Legendre
    // polynomial expansion does not converge, check the whole list

    if (fmRemoteSPIndex == -1 || rr>fContainer.GetParameters().GetConvergenceRatio())
    {
      rrmin = 1.e20;

      for (unsigned int i=0;i<fContainer.GetRemoteSourcePoints().size();i++)
      {
	const KZonalHarmonicSourcePoint& sP = *(fContainer.GetRemoteSourcePoints().at(i));

	delz = z-sP.GetZ0();
	rho  = sqrt(r*r+delz*delz);
	rr=sP.GetRho()/rho;

	if (rr<rrmin)
	{
	  rrmin=rr;
	  fmRemoteSPIndex = i;
	}
      }
    }

    if (rrmin>fContainer.GetParameters().GetConvergenceRatio())
      return false;
    else
      return true;
  }
} // end namespace KEMField

#endif /* KZONALHARMONICCOMPUTER_DEF */
