#ifndef KELECTROMAGNETINTEGRATINGFIELDSOLVER_DEF
#define KELECTROMAGNETINTEGRATINGFIELDSOLVER_DEF

#include "KElectromagnetContainer.hh"
#include "KIntegratingFieldSolverTemplate.hh"
#include "KThreeMatrix_KEMField.hh"

namespace KEMField
{
class ElectromagnetSingleThread;

template<class Integrator> class KIntegratingFieldSolver<Integrator, ElectromagnetSingleThread>
{
  public:
    KIntegratingFieldSolver(KElectromagnetContainer& container, Integrator& integrator) :
        fContainer(container),
        fIntegrator(integrator)
    {}
    virtual ~KIntegratingFieldSolver() = default;

    KFieldVector VectorPotential(const KPosition& P) const;
    KFieldVector MagneticField(const KPosition& P) const;
    KGradient MagneticFieldGradient(const KPosition& P) const;

    const KElectromagnetContainer& GetContainer() const
    {
        return fContainer;
    }

  protected:
    class VectorPotentialAction
    {
      public:
        VectorPotentialAction(const KElectromagnetContainer& container, const Integrator& integrator,
                              const KPosition& P) :
            fContainer(container),
            fIntegrator(integrator),
            fP(P),
            fVectorPotential(0., 0., 0.)
        {}
        ~VectorPotentialAction() = default;

        KFieldVector GetVectorPotential() const
        {
            return fVectorPotential;
        }

        template<class Electromagnet> void Act(Type2Type<Electromagnet>)
        {
            typename std::vector<Electromagnet*>::const_iterator it;
            for (it = fContainer.Vector<Electromagnet>().begin(); it != fContainer.Vector<Electromagnet>().end(); ++it)
                fVectorPotential += fIntegrator.VectorPotential(*(*it), fP);
        }

      private:
        const KElectromagnetContainer& fContainer;
        const Integrator& fIntegrator;
        const KPosition& fP;
        KFieldVector fVectorPotential;
    };

    class MagneticFieldAction
    {
      public:
        MagneticFieldAction(const KElectromagnetContainer& container, const Integrator& integrator,
                            const KPosition& P) :
            fContainer(container),
            fIntegrator(integrator),
            fP(P),
            fMagneticField(0., 0., 0.)
        {}
        ~MagneticFieldAction() = default;

        KFieldVector GetMagneticField() const
        {
            return fMagneticField;
        }

        template<class Electromagnet> void Act(Type2Type<Electromagnet>)
        {
            typename std::vector<Electromagnet*>::const_iterator it;
            for (it = fContainer.Vector<Electromagnet>().begin(); it != fContainer.Vector<Electromagnet>().end(); ++it)
                fMagneticField += fIntegrator.MagneticField(*(*it), fP);
        }

      private:
        const KElectromagnetContainer& fContainer;
        const Integrator& fIntegrator;
        const KPosition& fP;
        KFieldVector fMagneticField;
    };

    KElectromagnetContainer& fContainer;
    Integrator& fIntegrator;
};


template<class Integrator>
KFieldVector KIntegratingFieldSolver<Integrator, ElectromagnetSingleThread>::VectorPotential(const KPosition& P) const
{
    VectorPotentialAction action(fContainer, fIntegrator, P);

    KElectromagnetAction<>::ActOnElectromagnets(action);

    return action.GetVectorPotential();
}

template<class Integrator>
KFieldVector KIntegratingFieldSolver<Integrator, ElectromagnetSingleThread>::MagneticField(const KPosition& P) const
{
    MagneticFieldAction action(fContainer, fIntegrator, P);

    KElectromagnetAction<>::ActOnElectromagnets(action);

    return action.GetMagneticField();
}

template<class Integrator>
KGradient
KIntegratingFieldSolver<Integrator, ElectromagnetSingleThread>::MagneticFieldGradient(const KPosition& P) const
{
    KGradient g;
    double epsilon = 1.e-6;
    for (unsigned int i = 0; i < 3; i++) {
        KPosition Pplus = P;
        Pplus[i] += epsilon;
        KPosition Pminus = P;
        Pminus[i] -= epsilon;
        KFieldVector Bplus = MagneticField(Pplus);
        KFieldVector Bminus = MagneticField(Pminus);
        for (unsigned int j = 0; j < 3; j++)
            g[j + 3 * i] = (Bplus[j] - Bminus[j]) / (2. * epsilon);
    }
    return g;
}

}  // namespace KEMField

#endif /* KELECTROMAGNETINTEGRATINGFIELDSOLVER_DEF */
