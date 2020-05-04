#ifndef KELECTROSTATICINTEGRATINGFIELDSOLVER_DEF
#define KELECTROSTATICINTEGRATINGFIELDSOLVER_DEF

#include "KIntegratingFieldSolverTemplate.hh"
#include "KSurfaceContainer.hh"

namespace KEMField
{
class ElectrostaticSingleThread;

template<class Integrator> class KIntegratingFieldSolver<Integrator, ElectrostaticSingleThread>
{
  public:
    typedef typename Integrator::Basis Basis;

    KIntegratingFieldSolver(const KSurfaceContainer& container, Integrator& integrator);
    virtual ~KIntegratingFieldSolver() {}

    virtual void Initialize() {}

    double Potential(const KPosition& P) const;
    KThreeVector ElectricField(const KPosition& P) const;
    std::pair<KThreeVector, double> ElectricFieldAndPotential(const KPosition& P) const;

    // functions without Kahan summation
    double PotentialNoKahanSum(const KPosition& P) const;
    KThreeVector ElectricFieldNoKahanSum(const KPosition& P) const;
    std::pair<KThreeVector, double> ElectricFieldAndPotentialNoKahanSum(const KPosition& P) const;

    double Potential(const unsigned int* SurfaceIndexSet, unsigned int SetSize, const KPosition& P) const;
    KThreeVector ElectricField(const unsigned int* SurfaceIndexSet, unsigned int SetSize, const KPosition& P) const;
    std::pair<KThreeVector, double> ElectricFieldAndPotential(const unsigned int* SurfaceIndexSet, unsigned int SetSize,
                                                              const KPosition& P) const;

  protected:
    const KSurfaceContainer& fContainer;
    Integrator& fIntegrator;

  private:
    class ShapeVisitorForPotential : public KSelectiveVisitor<KShapeVisitor, typename Integrator::AcceptedShapes>
    {
      public:
        using KSelectiveVisitor<KShapeVisitor, typename Integrator::AcceptedShapes>::Visit;

        ShapeVisitorForPotential(Integrator& integrator) : fIntegrator(integrator) {}

        void Visit(KTriangle& t) override
        {
            ComputePotential(t);
        }
        void Visit(KRectangle& r) override
        {
            ComputePotential(r);
        }
        void Visit(KLineSegment& l) override
        {
            ComputePotential(l);
        }
        void Visit(KConicSection& c) override
        {
            ComputePotential(c);
        }
        void Visit(KRing& r) override
        {
            ComputePotential(r);
        }
        void Visit(KTriangleGroup& t) override
        {
            ComputePotential(t);
        }
        void Visit(KRectangleGroup& r) override
        {
            ComputePotential(r);
        }
        void Visit(KLineSegmentGroup& l) override
        {
            ComputePotential(l);
        }
        void Visit(KConicSectionGroup& c) override
        {
            ComputePotential(c);
        }
        void Visit(KRingGroup& r) override
        {
            ComputePotential(r);
        }

        template<class ShapePolicy> void ComputePotential(ShapePolicy& s)
        {
            fPotential = fIntegrator.Potential(&s, fP);
        }

        void SetPosition(const KPosition& p) const
        {
            fP = p;
        }
        double GetNormalizedPotential() const
        {
            return fPotential;
        }

      private:
        mutable KPosition fP;
        double fPotential;
        Integrator& fIntegrator;
    };

    class ShapeVisitorForElectricField : public KSelectiveVisitor<KShapeVisitor, typename Integrator::AcceptedShapes>
    {
      public:
        using KSelectiveVisitor<KShapeVisitor, typename Integrator::AcceptedShapes>::Visit;

        ShapeVisitorForElectricField(Integrator& integrator) : fIntegrator(integrator) {}

        void Visit(KTriangle& t) override
        {
            ComputeElectricField(t);
        }
        void Visit(KRectangle& r) override
        {
            ComputeElectricField(r);
        }
        void Visit(KLineSegment& l) override
        {
            ComputeElectricField(l);
        }
        void Visit(KConicSection& c) override
        {
            ComputeElectricField(c);
        }
        void Visit(KRing& r) override
        {
            ComputeElectricField(r);
        }
        void Visit(KTriangleGroup& t) override
        {
            ComputeElectricField(t);
        }
        void Visit(KRectangleGroup& r) override
        {
            ComputeElectricField(r);
        }
        void Visit(KLineSegmentGroup& l) override
        {
            ComputeElectricField(l);
        }
        void Visit(KConicSectionGroup& c) override
        {
            ComputeElectricField(c);
        }
        void Visit(KRingGroup& r) override
        {
            ComputeElectricField(r);
        }

        template<class ShapePolicy> void ComputeElectricField(ShapePolicy& s)
        {
            fElectricField = fIntegrator.ElectricField(&s, fP);
        }

        void SetPosition(const KPosition& p) const
        {
            fP = p;
        }
        KThreeVector& GetNormalizedElectricField() const
        {
            return fElectricField;
        }

      private:
        mutable KPosition fP;
        mutable KThreeVector fElectricField;
        Integrator& fIntegrator;
    };

    class ShapeVisitorForElectricFieldAndPotential :
        public KSelectiveVisitor<KShapeVisitor, typename Integrator::AcceptedShapes>
    {
      public:
        using KSelectiveVisitor<KShapeVisitor, typename Integrator::AcceptedShapes>::Visit;

        ShapeVisitorForElectricFieldAndPotential(Integrator& integrator) : fIntegrator(integrator) {}

        void Visit(KTriangle& t) override
        {
            ComputeElectricFieldAndPotential(t);
        }
        void Visit(KRectangle& r) override
        {
            ComputeElectricFieldAndPotential(r);
        }
        void Visit(KLineSegment& l) override
        {
            ComputeElectricFieldAndPotential(l);
        }
        void Visit(KConicSection& c) override
        {
            ComputeElectricFieldAndPotential(c);
        }
        void Visit(KRing& r) override
        {
            ComputeElectricFieldAndPotential(r);
        }
        void Visit(KTriangleGroup& t) override
        {
            ComputeElectricFieldAndPotential(t);
        }
        void Visit(KRectangleGroup& r) override
        {
            ComputeElectricFieldAndPotential(r);
        }
        void Visit(KLineSegmentGroup& l) override
        {
            ComputeElectricFieldAndPotential(l);
        }
        void Visit(KConicSectionGroup& c) override
        {
            ComputeElectricFieldAndPotential(c);
        }
        void Visit(KRingGroup& r) override
        {
            ComputeElectricFieldAndPotential(r);
        }

        template<class ShapePolicy> void ComputeElectricFieldAndPotential(ShapePolicy& s)
        {
            fElectricFieldAndPotential = fIntegrator.ElectricFieldAndPotential(&s, fP);
        }

        void SetPosition(const KPosition& p) const
        {
            fP = p;
        }
        std::pair<KThreeVector, double>& GetNormalizedElectricFieldAndPotential() const
        {
            return fElectricFieldAndPotential;
        }

      private:
        mutable KPosition fP;
        mutable std::pair<KThreeVector, double> fElectricFieldAndPotential;
        Integrator& fIntegrator;
    };

    mutable ShapeVisitorForPotential fShapeVisitorForPotential;
    mutable ShapeVisitorForElectricField fShapeVisitorForElectricField;
    mutable ShapeVisitorForElectricFieldAndPotential fShapeVisitorForElectricFieldAndPotential;
};


template<class Integrator>
KIntegratingFieldSolver<Integrator, ElectrostaticSingleThread>::KIntegratingFieldSolver(
    const KSurfaceContainer& container, Integrator& integrator) :
    fContainer(container),
    fIntegrator(integrator),
    fShapeVisitorForPotential(integrator),
    fShapeVisitorForElectricField(integrator),
    fShapeVisitorForElectricFieldAndPotential(integrator)
{}

template<class Integrator>
double KIntegratingFieldSolver<Integrator, ElectrostaticSingleThread>::Potential(const KPosition& P) const
{
    // Kahan Sum to mitigate rounding error
    fShapeVisitorForPotential.SetPosition(P);
    double sum = 0.;
    double c = 0.;
    double y = 0.;
    double t = 0.;
    KSurfaceContainer::iterator it;
    for (it = fContainer.begin<Basis>(); it != fContainer.end<Basis>(); ++it) {
        (*it)->Accept(fShapeVisitorForPotential);
        y = fShapeVisitorForPotential.GetNormalizedPotential() * fIntegrator.BasisValue(*it, 0) - c;
        t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}

template<class Integrator>
KThreeVector KIntegratingFieldSolver<Integrator, ElectrostaticSingleThread>::ElectricField(const KPosition& P) const
{
    // Kahan Sum to mitigate rounding error
    fShapeVisitorForElectricField.SetPosition(P);
    KThreeVector sum(0., 0., 0.);
    KThreeVector c(0., 0., 0.);
    KThreeVector y(0., 0., 0.);
    KThreeVector t(0., 0., 0.);
    KSurfaceContainer::iterator it;
    for (it = fContainer.begin<Basis>(); it != fContainer.end<Basis>(); ++it) {
        (*it)->Accept(fShapeVisitorForElectricField);
        y = fShapeVisitorForElectricField.GetNormalizedElectricField() * fIntegrator.BasisValue(*it, 0) - c;
        t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}

template<class Integrator>
std::pair<KThreeVector, double>
KIntegratingFieldSolver<Integrator, ElectrostaticSingleThread>::ElectricFieldAndPotential(const KPosition& P) const
{
    // Kahan Sum to mitigate rounding error
    fShapeVisitorForElectricFieldAndPotential.SetPosition(P);

    KThreeVector sumField(0., 0., 0.);
    KThreeVector cField(0., 0., 0.);
    KThreeVector yField(0., 0., 0.);
    KThreeVector tField(0., 0., 0.);

    double sumPot = 0.;
    double cPot = 0.;
    double yPot = 0.;
    double tPot = 0.;

    KSurfaceContainer::iterator it;
    std::pair<KThreeVector, double> itFieldAndPot;
    double itBasisValue = 0.;

    for (it = fContainer.begin<Basis>(); it != fContainer.end<Basis>(); ++it) {
        (*it)->Accept(fShapeVisitorForElectricFieldAndPotential);

        itFieldAndPot = fShapeVisitorForElectricFieldAndPotential.GetNormalizedElectricFieldAndPotential();
        itBasisValue = fIntegrator.BasisValue(*it, 0);

        yField = itFieldAndPot.first * itBasisValue - cField;
        tField = sumField + yField;
        cField = (tField - sumField) - yField;
        sumField = tField;

        yPot = itFieldAndPot.second * itBasisValue - cPot;
        tPot = sumPot + yPot;
        cPot = (tPot - sumPot) - yPot;
        sumPot = tPot;
    }
    return std::make_pair(sumField, sumPot);
}

template<class Integrator>
double KIntegratingFieldSolver<Integrator, ElectrostaticSingleThread>::PotentialNoKahanSum(const KPosition& P) const
{
    fShapeVisitorForPotential.SetPosition(P);
    double sum = 0.;
    KSurfaceContainer::iterator it;
    for (it = fContainer.begin<Basis>(); it != fContainer.end<Basis>(); ++it) {
        (*it)->Accept(fShapeVisitorForPotential);
        sum += (fShapeVisitorForPotential.GetNormalizedPotential() * fIntegrator.BasisValue(*it, 0));
    }
    return sum;
}

template<class Integrator>
KThreeVector
KIntegratingFieldSolver<Integrator, ElectrostaticSingleThread>::ElectricFieldNoKahanSum(const KPosition& P) const
{
    // Kahan Sum to mitigate rounding error
    fShapeVisitorForElectricField.SetPosition(P);
    KThreeVector sum(0., 0., 0.);
    KSurfaceContainer::iterator it;
    for (it = fContainer.begin<Basis>(); it != fContainer.end<Basis>(); ++it) {
        (*it)->Accept(fShapeVisitorForElectricField);
        sum += (fShapeVisitorForElectricField.GetNormalizedElectricField() * fIntegrator.BasisValue(*it, 0));
    }
    return sum;
}

template<class Integrator>
std::pair<KThreeVector, double>
KIntegratingFieldSolver<Integrator, ElectrostaticSingleThread>::ElectricFieldAndPotentialNoKahanSum(
    const KPosition& P) const
{
    // Kahan Sum to mitigate rounding error
    fShapeVisitorForElectricFieldAndPotential.SetPosition(P);

    KThreeVector sumField(0., 0., 0.);
    double sumPot = 0.;

    KSurfaceContainer::iterator it;
    std::pair<KThreeVector, double> itFieldAndPot;
    double itBasisValue = 0.;

    for (it = fContainer.begin<Basis>(); it != fContainer.end<Basis>(); ++it) {
        (*it)->Accept(fShapeVisitorForElectricFieldAndPotential);

        itFieldAndPot = fShapeVisitorForElectricFieldAndPotential.GetNormalizedElectricFieldAndPotential();
        itBasisValue = fIntegrator.BasisValue(*it, 0);

        sumField += (itFieldAndPot.first * itBasisValue);
        sumPot += (itFieldAndPot.second * itBasisValue);
    }
    return std::make_pair(sumField, sumPot);
}

template<class Integrator>
double KIntegratingFieldSolver<Integrator, ElectrostaticSingleThread>::Potential(const unsigned int* SurfaceIndexSet,
                                                                                 unsigned int SetSize,
                                                                                 const KPosition& P) const
{
    // Kahan Sum to mitigate rounding error
    fShapeVisitorForPotential.SetPosition(P);
    double sum = 0.;
    double c = 0.;
    double y = 0.;
    double t = 0.;
    unsigned int id;
    for (unsigned int i = 0; i < SetSize; ++i) {
        id = SurfaceIndexSet[i];
        fContainer[id]->Accept(fShapeVisitorForPotential);
        y = fShapeVisitorForPotential.GetNormalizedPotential() * fIntegrator.BasisValue(fContainer[id], 0) - c;
        t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}

template<class Integrator>
KThreeVector KIntegratingFieldSolver<Integrator, ElectrostaticSingleThread>::ElectricField(
    const unsigned int* SurfaceIndexSet, unsigned int SetSize, const KPosition& P) const
{
    // Kahan Sum to mitigate rounding error
    fShapeVisitorForElectricField.SetPosition(P);
    KThreeVector sum(0., 0., 0.);
    KThreeVector c(0., 0., 0.);
    KThreeVector y(0., 0., 0.);
    KThreeVector t(0., 0., 0.);
    unsigned int id;
    for (unsigned int i = 0; i < SetSize; ++i) {
        id = SurfaceIndexSet[i];
        fContainer[id]->Accept(fShapeVisitorForElectricField);
        y = fShapeVisitorForElectricField.GetNormalizedElectricField() * fIntegrator.BasisValue(fContainer[id], 0) - c;
        t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}

template<class Integrator>
std::pair<KThreeVector, double>
KIntegratingFieldSolver<Integrator, ElectrostaticSingleThread>::ElectricFieldAndPotential(
    const unsigned int* SurfaceIndexSet, unsigned int SetSize, const KPosition& P) const
{
    // Kahan Sum to mitigate rounding error
    fShapeVisitorForElectricFieldAndPotential.SetPosition(P);

    KThreeVector sumField(0., 0., 0.);
    KThreeVector cField(0., 0., 0.);
    KThreeVector yField(0., 0., 0.);
    KThreeVector tField(0., 0., 0.);

    double sumPot = 0.;
    double cPot = 0.;
    double yPot = 0.;
    double tPot = 0.;

    KSurfaceContainer::iterator it;
    std::pair<KThreeVector, double> itFieldAndPot;
    double itBasisValue = 0.;

    unsigned int id;

    for (unsigned int i = 0; i < SetSize; ++i) {
        id = SurfaceIndexSet[i];
        fContainer[id]->Accept(fShapeVisitorForElectricFieldAndPotential);

        itFieldAndPot = fShapeVisitorForElectricFieldAndPotential.GetNormalizedElectricFieldAndPotential();
        itBasisValue = fIntegrator.BasisValue(*it, 0);

        yField = itFieldAndPot.first * itBasisValue - cField;
        tField = sumField + yField;
        cField = (tField - sumField) - yField;
        sumField = tField;

        yPot = itFieldAndPot.second * itBasisValue - cPot;
        tPot = sumPot + yPot;
        cPot = (tPot - sumPot) - yPot;
        sumPot = tPot;
    }
    return std::make_pair(sumField, sumPot);
}

}  // namespace KEMField

#endif /* KELECTROSTATICINTEGRATINGFIELDSOLVER_DEF */
