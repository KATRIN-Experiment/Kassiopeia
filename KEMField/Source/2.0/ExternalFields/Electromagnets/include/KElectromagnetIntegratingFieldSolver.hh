#ifndef KELECTROMAGNETINTEGRATINGFIELDSOLVER_DEF
#define KELECTROMAGNETINTEGRATINGFIELDSOLVER_DEF

#include "KEMThreeMatrix.hh"

#include "KElectromagnetContainer.hh"
#include "KElectromagnetIntegrator.hh"

namespace KEMField
{
  template <class Integrator>
  class KIntegratingFieldSolver;

  template <>
  class KIntegratingFieldSolver<KElectromagnetIntegrator>
  {
  public:
    KIntegratingFieldSolver(KElectromagnetContainer& container,
			    KElectromagnetIntegrator& integrator)
      : fContainer(container),
	fIntegrator(integrator) {}
    virtual ~KIntegratingFieldSolver() {}

    KEMThreeVector VectorPotential(const KPosition& P) const;
    KEMThreeVector MagneticField(const KPosition& P) const;
    KGradient MagneticFieldGradient(const KPosition& P) const;

    const KElectromagnetContainer& GetContainer() const { return fContainer; }

  protected:
    class VectorPotentialAction
    {
    public:
      VectorPotentialAction(const KElectromagnetContainer& container,
			    const KElectromagnetIntegrator& integrator,
			    const KPosition& P) :
	fContainer(container),fIntegrator(integrator),fP(P),fVectorPotential(0.,0.,0.) {}
      ~VectorPotentialAction() {}

      KEMThreeVector GetVectorPotential() const { return fVectorPotential; }

      template <class Electromagnet>
      void Act(Type2Type<Electromagnet>)
      {
	typename std::vector<Electromagnet*>::const_iterator it;
	for (it=fContainer.Vector<Electromagnet>().begin();it!=fContainer.Vector<Electromagnet>().end();++it)
	  fVectorPotential += fIntegrator.VectorPotential(*(*it),fP);
      }

    private:
      const KElectromagnetContainer& fContainer;
      const KElectromagnetIntegrator& fIntegrator;
      const KPosition& fP;
      KEMThreeVector fVectorPotential;
    };

    class MagneticFieldAction
    {
    public:
      MagneticFieldAction(const KElectromagnetContainer& container,
			  const KElectromagnetIntegrator& integrator,
			  const KPosition& P) :
	fContainer(container),fIntegrator(integrator),fP(P),fMagneticField(0.,0.,0.) {}
      ~MagneticFieldAction() {}

      KEMThreeVector GetMagneticField() const { return fMagneticField; }

      template <class Electromagnet>
      void Act(Type2Type<Electromagnet>)
      {
	typename std::vector<Electromagnet*>::const_iterator it;
	for (it=fContainer.Vector<Electromagnet>().begin();it!=fContainer.Vector<Electromagnet>().end();++it)
	  fMagneticField += fIntegrator.MagneticField(*(*it),fP);
      }

    private:
      const KElectromagnetContainer& fContainer;
      const KElectromagnetIntegrator& fIntegrator;
      const KPosition& fP;
      KEMThreeVector fMagneticField;
    };

    KElectromagnetContainer& fContainer;
    KElectromagnetIntegrator& fIntegrator;
  };
}

#endif /* KELECTROMAGNETINTEGRATINGFIELDSOLVER_DEF */
