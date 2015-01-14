#ifndef KELECTROSTATICZONALHARMONICFIELDSOLVER_DEF
#define KELECTROSTATICZONALHARMONICFIELDSOLVER_DEF

#include "KEMThreeVector.hh"

#include "KZonalHarmonicComputer.hh"

#include "KElectrostaticIntegratingFieldSolver.hh"

namespace KEMField
{
  template <class Basis>
  class KZonalHarmonicFieldSolver;

  template <>
  class KZonalHarmonicFieldSolver<KElectrostaticBasis> :
    public KZonalHarmonicComputer<KElectrostaticBasis>
  {
  public:
    typedef KZonalHarmonicContainer<KElectrostaticBasis> Container;
    typedef KZonalHarmonicTrait<KElectrostaticBasis>::Integrator Integrator;

    KZonalHarmonicFieldSolver(Container& container,
			      Integrator& integrator) :
      KZonalHarmonicComputer<KElectrostaticBasis>(container,integrator),
      fIntegratingFieldSolver(container.GetElementContainer(),integrator) {}

    virtual ~KZonalHarmonicFieldSolver() {}

    double Potential(const KPosition& P) const;
    KEMThreeVector ElectricField(const KPosition& P) const;

  private:
    bool CentralExpansion(const KPosition& P,
			    double& potential,
			    KEMThreeVector& electricField) const;
    bool RemoteExpansion(const KPosition& P,
			   double& potential,
			   KEMThreeVector& electricField) const;

    KIntegratingFieldSolver<Integrator> fIntegratingFieldSolver;

    class PotentialAccumulator
    {
    public:
      PotentialAccumulator(const KPosition& P) : fP(P) {}
      double operator()(double phi,
			  KZonalHarmonicFieldSolver<KElectrostaticBasis>* c)
      {
	return phi + c->Potential(fP);
      }

    private:
      const KPosition& fP;
    };

    class ElectricFieldAccumulator
    {
    public:
      ElectricFieldAccumulator(const KPosition& P) : fP(P) {}
      KEMThreeVector operator()(KEMThreeVector electricField,
			  KZonalHarmonicFieldSolver<KElectrostaticBasis>* c)
      {
	return electricField + c->ElectricField(fP);
      }

    private:
      const KPosition& fP;
    };
  };

} // end namespace KEMField

#include "KZonalHarmonicComputer.icc"

#endif /* KELECTROSTATICZONALHARMONICFIELDSOLVER_DEF */
