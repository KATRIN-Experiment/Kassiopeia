#ifndef KELECTROMAGNETZONALHARMONICFIELDSOLVER_DEF
#define KELECTROMAGNETZONALHARMONICFIELDSOLVER_DEF

#include "KEMThreeVector.hh"
#include "KEMThreeMatrix.hh"

#include "KZonalHarmonicComputer.hh"

#include "KElectromagnetIntegratingFieldSolver.hh"

namespace KEMField
{
  template <class Basis>
  class KZonalHarmonicFieldSolver;

  template <>
  class KZonalHarmonicFieldSolver<KMagnetostaticBasis> :
    public KZonalHarmonicComputer<KMagnetostaticBasis>
  {
  public:
    typedef KZonalHarmonicContainer<KMagnetostaticBasis> Container;
    typedef KZonalHarmonicTrait<KMagnetostaticBasis>::Integrator Integrator;

    KZonalHarmonicFieldSolver(Container& container,
			      Integrator& integrator) :
      KZonalHarmonicComputer<KMagnetostaticBasis>(container,integrator),
      fIntegratingFieldSolver(container.GetElementContainer(),integrator) {}

    virtual ~KZonalHarmonicFieldSolver() {}

    KEMThreeVector VectorPotential(const KPosition& P) const;
    KEMThreeVector MagneticField(const KPosition& P) const;
    KGradient MagneticFieldGradient(const KPosition& P) const;

  private:
    bool CentralExpansion(const KPosition& P,
			    KEMThreeVector& A,
			    KEMThreeVector& B) const;
    bool RemoteExpansion(const KPosition& P,
			   KEMThreeVector& A,
			   KEMThreeVector& B) const;
    bool CentralGradientExpansion(const KPosition& P,
				  KGradient& g) const;
    bool RemoteGradientExpansion(const KPosition& P,
				 KGradient& g) const;

    KIntegratingFieldSolver<Integrator> fIntegratingFieldSolver;

    class VectorPotentialAccumulator
    {
    public:
      VectorPotentialAccumulator(const KPosition& P) : fP(P) {}
      KEMThreeVector operator()(KEMThreeVector vectorPotential,
			      KZonalHarmonicFieldSolver<KMagnetostaticBasis>* c)
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
      KEMThreeVector operator()(KEMThreeVector magneticField,
			      KZonalHarmonicFieldSolver<KMagnetostaticBasis>* c)
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
      KEMThreeMatrix operator()(KEMThreeMatrix magneticFieldGradient,
			   KZonalHarmonicFieldSolver<KMagnetostaticBasis>* c)
      {
	return magneticFieldGradient + c->MagneticFieldGradient(fP);
      }

    private:
      const KPosition& fP;
    };
  };

} // end namespace KEMField

#include "KZonalHarmonicComputer.icc"

#endif /* KELECTROMAGNETZONALHARMONICFIELDSOLVER_DEF */
