#ifndef KELECTROMAGNETZONALHARMONICFIELDSOLVER_DEF
#define KELECTROMAGNETZONALHARMONICFIELDSOLVER_DEF

#include "KElectromagnetIntegratingFieldSolver.hh"
#include "KThreeMatrix_KEMField.hh"
#include "KThreeVector_KEMField.hh"
#include "KZonalHarmonicComputer.hh"

#include "KThreeMatrix.hh"

namespace KEMField
{
template<class Basis> class KZonalHarmonicFieldSolver;

template<> class KZonalHarmonicFieldSolver<KMagnetostaticBasis> : public KZonalHarmonicComputer<KMagnetostaticBasis>
{
  public:
    using Container = KZonalHarmonicContainer<KMagnetostaticBasis>;
    using Integrator = KZonalHarmonicTrait<KMagnetostaticBasis>::Integrator;

    KZonalHarmonicFieldSolver(Container& container, Integrator& integrator) :
        KZonalHarmonicComputer<KMagnetostaticBasis>(container, integrator),
        fIntegratingFieldSolver(container.GetElementContainer(), integrator)
    {
        fZHCoeffSingleton = KZHLegendreCoefficients::GetInstance();
    }

    ~KZonalHarmonicFieldSolver() override = default;

    KFieldVector VectorPotential(const KPosition& P) const;
    KFieldVector MagneticField(const KPosition& P) const;
    KGradient MagneticFieldGradient(const KPosition& P) const;
    std::pair<KFieldVector, KGradient> MagneticFieldAndGradient(const KPosition& P) const;

    bool CentralExpansion(const KPosition& P) const;
    bool RemoteExpansion(const KPosition& P) const;

  private:
    KZHLegendreCoefficients* fZHCoeffSingleton;

    bool CentralExpansionMagneticField(const KPosition& P, KFieldVector& B) const;
    bool CentralExpansionVectorPotential(const KPosition& P, KFieldVector& A) const;
    bool RemoteExpansionMagneticField(const KPosition& P, KFieldVector& B) const;
    bool RemoteExpansionVectorPotential(const KPosition& P, KFieldVector& A) const;
    bool CentralGradientExpansion(const KPosition& P, KGradient& g) const;
    bool RemoteGradientExpansion(const KPosition& P, KGradient& g) const;
    bool CentralGradientExpansionNumerical(const KPosition& P, KGradient& g) const;
    bool RemoteGradientExpansionNumerical(const KPosition& P, KGradient& g) const;
    bool CentralMagneticFieldAndGradientExpansion(const KPosition& P, KGradient& g, KFieldVector& B) const;
    bool RemoteMagneticFieldAndGradientExpansion(const KPosition& P, KGradient& g, KFieldVector& B) const;

    KIntegratingFieldSolver<Integrator> fIntegratingFieldSolver;

    class VectorPotentialAccumulator
    {
      public:
        VectorPotentialAccumulator(const KPosition& P) : fP(P) {}
        KFieldVector operator()(const KFieldVector& vectorPotential, KZonalHarmonicFieldSolver<KMagnetostaticBasis>* c)
        {
            return vectorPotential + c->VectorPotential(fP);
        }

      private:
        const KPosition& fP;
    };

    class MagneticFieldAccumulator
    {
      public:
        MagneticFieldAccumulator(const KPosition& P) : fP(P) {}
        KFieldVector operator()(const KFieldVector& magneticField, KZonalHarmonicFieldSolver<KMagnetostaticBasis>* c)
        {
            return magneticField + c->MagneticField(fP);
        }

      private:
        const KPosition& fP;
    };

    class MagneticFieldGradientAccumulator
    {
      public:
        MagneticFieldGradientAccumulator(const KPosition& P) : fP(P) {}
        katrin::KThreeMatrix operator()(const katrin::KThreeMatrix& magneticFieldGradient,
                                KZonalHarmonicFieldSolver<KMagnetostaticBasis>* c)
        {
            return magneticFieldGradient + c->MagneticFieldGradient(fP);
        }

      private:
        const KPosition& fP;
    };

    class MagneticFieldAndGradientAccumulator
    {
      public:
        MagneticFieldAndGradientAccumulator(const KPosition& P) : fP(P) {}
        std::pair<KFieldVector, katrin::KThreeMatrix>
        operator()(const std::pair<KFieldVector, katrin::KThreeMatrix>& fieldandgradient,
                   KZonalHarmonicFieldSolver<KMagnetostaticBasis>* c)
        {
            std::pair<KFieldVector, katrin::KThreeMatrix> pair = c->MagneticFieldAndGradient(fP);
            return std::make_pair(fieldandgradient.first + pair.first, fieldandgradient.second + pair.second);
        }

      private:
        const KPosition& fP;
    };
};

}  // end namespace KEMField

#include "KZonalHarmonicComputer.icc"

#endif /* KELECTROMAGNETZONALHARMONICFIELDSOLVER_DEF */
