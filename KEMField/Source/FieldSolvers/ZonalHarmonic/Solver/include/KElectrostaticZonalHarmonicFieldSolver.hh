#ifndef KELECTROSTATICZONALHARMONICFIELDSOLVER_DEF
#define KELECTROSTATICZONALHARMONICFIELDSOLVER_DEF

#include "KElectrostaticIntegratingFieldSolver.hh"
#include "KThreeVector_KEMField.hh"
#include "KZonalHarmonicComputer.hh"

namespace KEMField
{
template<class Basis> class KZonalHarmonicFieldSolver;

template<> class KZonalHarmonicFieldSolver<KElectrostaticBasis> : public KZonalHarmonicComputer<KElectrostaticBasis>
{
  public:
    typedef KZonalHarmonicContainer<KElectrostaticBasis> Container;
    typedef KZonalHarmonicTrait<KElectrostaticBasis>::Integrator Integrator;

    KZonalHarmonicFieldSolver(Container& container, Integrator& integrator) :
        KZonalHarmonicComputer<KElectrostaticBasis>(container, integrator),
        fIntegratingFieldSolver(container.GetElementContainer(), integrator)
    {
        fZHCoeffSingleton = KZHLegendreCoefficients::GetInstance();
    }

    ~KZonalHarmonicFieldSolver() override {}

    bool CentralExpansion(const KPosition& P) const;
    bool RemoteExpansion(const KPosition& P) const;

    double Potential(const KPosition& P) const;
    KThreeVector ElectricField(const KPosition& P) const;
    std::pair<KThreeVector, double> ElectricFieldAndPotential(const KPosition& P) const;


  private:
    KZHLegendreCoefficients* fZHCoeffSingleton;

    bool CentralExpansionPotential(const KPosition& P, double& potential) const;
    bool RemoteExpansionPotential(const KPosition& P, double& potential) const;

    bool CentralExpansionField(const KPosition& P, KThreeVector& electricField) const;
    bool RemoteExpansionField(const KPosition& P, KThreeVector& electricField) const;

    bool CentralExpansionFieldAndPotential(const KPosition& P, KThreeVector& electricField, double& potential) const;
    bool RemoteExpansionFieldAndPotential(const KPosition& P, KThreeVector& electricField, double& potential) const;

    KIntegratingFieldSolver<Integrator> fIntegratingFieldSolver;

    class PotentialAccumulator
    {
      public:
        PotentialAccumulator(const KPosition& P) : fP(P) {}
        double operator()(double phi, KZonalHarmonicFieldSolver<KElectrostaticBasis>* c)
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
        KThreeVector operator()(KThreeVector electricField, KZonalHarmonicFieldSolver<KElectrostaticBasis>* c)
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
        std::pair<KThreeVector, double> operator()(std::pair<KThreeVector, double> FieldandPotential,
                                                   KZonalHarmonicFieldSolver<KElectrostaticBasis>* c)
        {
            std::pair<KThreeVector, double> pair = c->ElectricFieldAndPotential(fP);
            return std::make_pair(FieldandPotential.first + pair.first, FieldandPotential.second + pair.second);
        }

      private:
        const KPosition& fP;
    };
};

}  // end namespace KEMField

#include "KZonalHarmonicComputer.icc"

#endif /* KELECTROSTATICZONALHARMONICFIELDSOLVER_DEF */
