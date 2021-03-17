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
    using Container = KZonalHarmonicContainer<KElectrostaticBasis>;
    using Integrator = KZonalHarmonicTrait<KElectrostaticBasis>::Integrator;

    KZonalHarmonicFieldSolver(Container& container, Integrator& integrator) :
        KZonalHarmonicComputer<KElectrostaticBasis>(container, integrator),
        fIntegratingFieldSolver(container.GetElementContainer(), integrator)
    {
        fZHCoeffSingleton = KZHLegendreCoefficients::GetInstance();
    }

    ~KZonalHarmonicFieldSolver() override = default;

    bool CentralExpansion(const KPosition& P) const;
    bool RemoteExpansion(const KPosition& P) const;

    double Potential(const KPosition& P) const;
    KFieldVector ElectricField(const KPosition& P) const;
    std::pair<KFieldVector, double> ElectricFieldAndPotential(const KPosition& P) const;


  private:
    KZHLegendreCoefficients* fZHCoeffSingleton;

    bool CentralExpansionPotential(const KPosition& P, double& potential) const;
    bool RemoteExpansionPotential(const KPosition& P, double& potential) const;

    bool CentralExpansionField(const KPosition& P, KFieldVector& electricField) const;
    bool RemoteExpansionField(const KPosition& P, KFieldVector& electricField) const;

    bool CentralExpansionFieldAndPotential(const KPosition& P, KFieldVector& electricField, double& potential) const;
    bool RemoteExpansionFieldAndPotential(const KPosition& P, KFieldVector& electricField, double& potential) const;

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
        KFieldVector operator()(const KFieldVector& electricField, KZonalHarmonicFieldSolver<KElectrostaticBasis>* c)
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
        std::pair<KFieldVector, double> operator()(const std::pair<KFieldVector, double>& FieldandPotential,
                                                   KZonalHarmonicFieldSolver<KElectrostaticBasis>* c)
        {
            std::pair<KFieldVector, double> pair = c->ElectricFieldAndPotential(fP);
            return std::make_pair(FieldandPotential.first + pair.first, FieldandPotential.second + pair.second);
        }

      private:
        const KPosition& fP;
    };
};

}  // end namespace KEMField

#include "KZonalHarmonicComputer.icc"

#endif /* KELECTROSTATICZONALHARMONICFIELDSOLVER_DEF */
