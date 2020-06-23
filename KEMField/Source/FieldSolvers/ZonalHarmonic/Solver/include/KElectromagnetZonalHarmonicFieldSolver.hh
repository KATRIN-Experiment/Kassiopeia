#ifndef KELECTROMAGNETZONALHARMONICFIELDSOLVER_DEF
#define KELECTROMAGNETZONALHARMONICFIELDSOLVER_DEF

#include "KElectromagnetIntegratingFieldSolver.hh"
#include "KThreeMatrix_KEMField.hh"
#include "KThreeVector_KEMField.hh"
#include "KZonalHarmonicComputer.hh"

namespace KEMField
{
template<class Basis> class KZonalHarmonicFieldSolver;

template<> class KZonalHarmonicFieldSolver<KMagnetostaticBasis> : public KZonalHarmonicComputer<KMagnetostaticBasis>
{
  public:
    typedef KZonalHarmonicContainer<KMagnetostaticBasis> Container;
    typedef KZonalHarmonicTrait<KMagnetostaticBasis>::Integrator Integrator;

    KZonalHarmonicFieldSolver(Container& container, Integrator& integrator) :
        KZonalHarmonicComputer<KMagnetostaticBasis>(container, integrator),
        fIntegratingFieldSolver(container.GetElementContainer(), integrator)
    {
        fZHCoeffSingleton = KZHLegendreCoefficients::GetInstance();
    }

    ~KZonalHarmonicFieldSolver() override {}

    KThreeVector VectorPotential(const KPosition& P) const;
    KThreeVector MagneticField(const KPosition& P) const;
    KGradient MagneticFieldGradient(const KPosition& P) const;
    std::pair<KThreeVector, KGradient> MagneticFieldAndGradient(const KPosition& P) const;

    bool CentralExpansion(const KPosition& P) const;
    bool RemoteExpansion(const KPosition& P) const;

  private:
    KZHLegendreCoefficients* fZHCoeffSingleton;

    bool CentralExpansionMagneticField(const KPosition& P, KThreeVector& B) const;
    bool CentralExpansionVectorPotential(const KPosition& P, KThreeVector& A) const;
    bool RemoteExpansionMagneticField(const KPosition& P, KThreeVector& B) const;
    bool RemoteExpansionVectorPotential(const KPosition& P, KThreeVector& A) const;
    bool CentralGradientExpansion(const KPosition& P, KGradient& g) const;
    bool RemoteGradientExpansion(const KPosition& P, KGradient& g) const;
    bool CentralGradientExpansionNumerical(const KPosition& P, KGradient& g) const;
    bool RemoteGradientExpansionNumerical(const KPosition& P, KGradient& g) const;
    bool CentralMagneticFieldAndGradientExpansion(const KPosition& P, KGradient& g, KThreeVector& B) const;
    bool RemoteMagneticFieldAndGradientExpansion(const KPosition& P, KGradient& g, KThreeVector& B) const;

    KIntegratingFieldSolver<Integrator> fIntegratingFieldSolver;

    class VectorPotentialAccumulator
    {
      public:
        VectorPotentialAccumulator(const KPosition& P) : fP(P) {}
        KThreeVector operator()(KThreeVector vectorPotential, KZonalHarmonicFieldSolver<KMagnetostaticBasis>* c)
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
        KThreeVector operator()(KThreeVector magneticField, KZonalHarmonicFieldSolver<KMagnetostaticBasis>* c)
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
        KThreeMatrix operator()(KThreeMatrix magneticFieldGradient, KZonalHarmonicFieldSolver<KMagnetostaticBasis>* c)
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
        std::pair<KThreeVector, KThreeMatrix> operator()(std::pair<KThreeVector, KThreeMatrix> fieldandgradient,
                                                         KZonalHarmonicFieldSolver<KMagnetostaticBasis>* c)
        {
            std::pair<KThreeVector, KThreeMatrix> pair = c->MagneticFieldAndGradient(fP);
            return std::make_pair(fieldandgradient.first + pair.first, fieldandgradient.second + pair.second);
        }

      private:
        const KPosition& fP;
    };
};

}  // end namespace KEMField

#include "KZonalHarmonicComputer.icc"

#endif /* KELECTROMAGNETZONALHARMONICFIELDSOLVER_DEF */
