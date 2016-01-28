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
      fIntegratingFieldSolver(container.GetElementContainer(),integrator)
      {
        fZHCoeffSingleton = KZHLegendreCoefficients::GetInstance();
      }

    virtual ~KZonalHarmonicFieldSolver() {}

    double Potential(const KPosition& P) const;
    KEMThreeVector ElectricField(const KPosition& P) const;
    std::pair<KEMThreeVector,double> ElectricFieldAndPotential(const KPosition& P) const;


  private:

    KZHLegendreCoefficients* fZHCoeffSingleton;

    bool CentralExpansionPotential(const KPosition& P, double& potential) const;
    bool RemoteExpansionPotential(const KPosition& P, double& potential) const;

    bool CentralExpansionField(const KPosition& P, KEMThreeVector& electricField) const;
    bool RemoteExpansionField(const KPosition& P, KEMThreeVector& electricField) const;

    bool CentralExpansionFieldAndPotential(const KPosition& P, KEMThreeVector& electricField, double& potential) const;
    bool RemoteExpansionFieldAndPotential(const KPosition& P, KEMThreeVector& electricField, double& potential) const;

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

    class ElectricFieldAndPotentialAccumulator
    {
    public:
      ElectricFieldAndPotentialAccumulator(const KPosition& P) : fP(P) {}
      std::pair<KEMThreeVector,double> operator()(std::pair<KEMThreeVector,double> FieldandPotential,
              KZonalHarmonicFieldSolver<KElectrostaticBasis>* c)
      {
          std::pair<KEMThreeVector, double> pair = c->ElectricFieldAndPotential(fP);
          return std::make_pair(FieldandPotential.first + pair.first, FieldandPotential.second + pair.second );
      }

    private:
      const KPosition& fP;
    };
  };

} // end namespace KEMField

#include "KZonalHarmonicComputer.icc"

#endif /* KELECTROSTATICZONALHARMONICFIELDSOLVER_DEF */
