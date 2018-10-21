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
      fmRemoteSPIndex(-1),
      fCentralFirst( true ){}

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
    mutable bool fCentralFirst;
  };

  template <class Basis>
  bool KZonalHarmonicComputer<Basis>::UseCentralExpansion(const KPosition& P) const
  {
    if (fContainer.GetCentralSourcePoints().empty()) return false;

    float p0 = (float)P[0];
    float p1 = (float)P[1];
    float z = (float)P[2];

    float r2 = p0*p0+p1*p1;


    float delz;
    float rc2 = 0;
    float rc2min = 1.e20;

    float convratiosquared = (float)fContainer.GetParameters().GetConvergenceRatio()*(float)fContainer.GetParameters().GetConvergenceRatio();

    // Check neighboring SP's if this is not the function's first call...
    if (fmCentralSPIndex != -1)
    {
        int lastSP = fmCentralSPIndex;

        for (int i=lastSP-2;i<=lastSP+2;i++)
        {
            if (i<0 || i>=(int)fContainer.GetCentralSourcePoints().size())
                continue;

            const KZonalHarmonicSourcePoint& sP = *(fContainer.GetCentralSourcePoints().at(i));

            delz = z-sP.GetFloatZ0();
            rc2 = (r2+delz*delz)*sP.Get1overRhosquared();

            if (rc2<rc2min)
            {
                rc2min=rc2;
                fmCentralSPIndex = i;
            }
        }


        if(rc2min > convratiosquared)
        {
            for (int k = 3; k<20; k++)
            {
                int i = lastSP - k;

                if (i > 0 && i < (int)fContainer.GetCentralSourcePoints().size())
                {

                    const KZonalHarmonicSourcePoint& sP = *(fContainer.GetCentralSourcePoints().at(i));

                    delz = z-sP.GetFloatZ0();
                    rc2 = (r2+delz*delz)*sP.Get1overRhosquared();

                    if (rc2<rc2min)
                    {
                        rc2min=rc2;
                        fmCentralSPIndex = i;
                    }
                }

                i = lastSP + k;

                if (i > 0 && i < (int)fContainer.GetCentralSourcePoints().size())
                {

                    const KZonalHarmonicSourcePoint& sP = *(fContainer.GetCentralSourcePoints().at(i));

                    delz = z-sP.GetFloatZ0();
                    rc2 = (r2+delz*delz)*sP.Get1overRhosquared();

                    if (rc2<rc2min)
                    {
                        rc2min=rc2;
                        fmCentralSPIndex = i;
                    }
                }
            }
        }

    }


    // ...if this is the function's first call, OR if Legendre
    // polynomial expansion does not converge, check the whole list

    if ( rc2min > convratiosquared || fmCentralSPIndex == -1)
    {
        rc2min = 1.e20;

        for (unsigned int i=0;i<fContainer.GetCentralSourcePoints().size();i++)
        {
            const KZonalHarmonicSourcePoint& sP = *(fContainer.GetCentralSourcePoints().at(i));

            delz = z-sP.GetFloatZ0();
            rc2 = (r2+delz*delz)*sP.Get1overRhosquared();

            if (rc2<rc2min)
            {
                rc2min=rc2;
                fmCentralSPIndex = i;
            }
        }
    }

    if (rc2min>convratiosquared)
    {
        fCentralFirst = false;
        return false;
    }
    else
    {
        fCentralFirst = true;
        return true;
    }
  }

  template <class Basis>
  bool KZonalHarmonicComputer<Basis>::UseRemoteExpansion(const KPosition& P) const
  {
    if (fContainer.GetRemoteSourcePoints().empty()) return false;

    float p0 = (float)P[0];
    float p1 = (float)P[1];
    float z = (float)P[2];

    float r2 = p0*p0+p1*p1;    
    float delz;

    float rr2 = 0;
    float rr2min = 1.e20;

    float convratiosquared = (float)fContainer.GetParameters().GetConvergenceRatio()*(float)fContainer.GetParameters().GetConvergenceRatio();

    // Check neighboring SP's if this is not the function's first call...
    if (fmRemoteSPIndex > 0 && fmRemoteSPIndex < (int)fContainer.GetRemoteSourcePoints().size() )
    {
        const KZonalHarmonicSourcePoint& sP = *(fContainer.GetRemoteSourcePoints().at(fmRemoteSPIndex));

        delz = z - sP.GetFloatZ0();
        rr2= sP.GetRhosquared()/(r2+delz*delz);

        if (rr2<rr2min)
        {
            rr2min=rr2;
        }
    }


    // ...if this is the function's first call, OR if Legendre
    // polynomial expansion does not converge, check the whole list

    if (fmRemoteSPIndex == -1 || rr2min > convratiosquared)
    {
        rr2min = 1.e20;

        for (unsigned int i=0;i<fContainer.GetRemoteSourcePoints().size();i++)
        {
            const KZonalHarmonicSourcePoint& sP = *(fContainer.GetRemoteSourcePoints().at(i));

            delz = z - sP.GetFloatZ0();
            rr2= sP.GetRhosquared()/(r2+delz*delz);

            if (rr2<rr2min)
            {
                rr2min=rr2;
                fmRemoteSPIndex = i;
            }
        }
    }

    if (rr2min > convratiosquared)
    {
        fCentralFirst = true;
        return false;
    }
    else
    {
        fCentralFirst = false;
        return true;
    }
  }
} // end namespace KEMField

#endif /* KZONALHARMONICCOMPUTER_DEF */
