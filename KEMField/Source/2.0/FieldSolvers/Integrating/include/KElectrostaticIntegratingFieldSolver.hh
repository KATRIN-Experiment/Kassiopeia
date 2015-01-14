#ifndef KINTEGRATINGFIELDSOLVER_DEF
#define KINTEGRATINGFIELDSOLVER_DEF

#include "KSurfaceContainer.hh"
#include "KElectrostaticBoundaryIntegrator.hh"

namespace KEMField
{
  template <class Integrator>
  class KIntegratingFieldSolver;

  template <>
  class KIntegratingFieldSolver<KElectrostaticBoundaryIntegrator>
  {
  public:
    typedef KElectrostaticBoundaryIntegrator::Basis Basis;

    KIntegratingFieldSolver(const KSurfaceContainer& container,
			    KElectrostaticBoundaryIntegrator& integrator);
    virtual ~KIntegratingFieldSolver() {}

    virtual void Initialize() {}

    double Potential(const KPosition& P) const;
    KEMThreeVector ElectricField(const KPosition& P) const;

    double Potential(const std::vector<unsigned int>* SurfaceIndexSet, const KPosition& P) const;
    KEMThreeVector ElectricField(const std::vector<unsigned int>* SurfaceIndexSet, const KPosition& P) const;

  protected:
    const KSurfaceContainer& fContainer;
    KElectrostaticBoundaryIntegrator& fIntegrator;

  private:
    class ShapeVisitorForPotential :
      public KSelectiveVisitor<KShapeVisitor,
			       KElectrostaticBoundaryIntegrator::AcceptedShapes>
    {
    public:
      using KSelectiveVisitor<KShapeVisitor,KElectrostaticBoundaryIntegrator::AcceptedShapes>::Visit;

     ShapeVisitorForPotential(KElectrostaticBoundaryIntegrator& integrator) :
      fIntegrator(integrator) {}

      void Visit(KTriangle& t) { ComputePotential(t); }
      void Visit(KRectangle& r) { ComputePotential(r); }
      void Visit(KLineSegment& l) { ComputePotential(l); }
      void Visit(KConicSection& c) { ComputePotential(c); }
      void Visit(KRing& r) { ComputePotential(r); }
      void Visit(KTriangleGroup& t) { ComputePotential(t); }
      void Visit(KRectangleGroup& r) { ComputePotential(r); }
      void Visit(KLineSegmentGroup& l) { ComputePotential(l);}
      void Visit(KConicSectionGroup& c) { ComputePotential(c); }
      void Visit(KRingGroup& r) { ComputePotential(r); }

      template <class ShapePolicy>
      void ComputePotential(ShapePolicy& s)
      {
	fPotential = fIntegrator.Potential(&s,fP);
      }

      void SetPosition(const KPosition& p) const { fP = p; }
      double GetNormalizedPotential() const { return fPotential; }

    private:
      mutable KPosition fP;
      double fPotential;
      KElectrostaticBoundaryIntegrator& fIntegrator;
    };

    class ShapeVisitorForElectricField :
      public KSelectiveVisitor<KShapeVisitor,
			       KElectrostaticBoundaryIntegrator::AcceptedShapes>
    {
    public:
      using KSelectiveVisitor<KShapeVisitor,KElectrostaticBoundaryIntegrator::AcceptedShapes>::Visit;

      ShapeVisitorForElectricField(KElectrostaticBoundaryIntegrator& integrator) : fIntegrator(integrator) {}

      void Visit(KTriangle& t) { ComputeElectricField(t); }
      void Visit(KRectangle& r) { ComputeElectricField(r); }
      void Visit(KLineSegment& l) { ComputeElectricField(l); }
      void Visit(KConicSection& c) { ComputeElectricField(c); }
      void Visit(KRing& r) { ComputeElectricField(r); }
      void Visit(KTriangleGroup& t) { ComputeElectricField(t); }
      void Visit(KRectangleGroup& r) { ComputeElectricField(r); }
      void Visit(KLineSegmentGroup& l) { ComputeElectricField(l);}
      void Visit(KConicSectionGroup& c) { ComputeElectricField(c); }
      void Visit(KRingGroup& r) { ComputeElectricField(r); }

      template <class ShapePolicy>
      void ComputeElectricField(ShapePolicy& s)
      {
	fElectricField = fIntegrator.ElectricField(&s,fP);
      }

      void SetPosition(const KPosition& p) const { fP = p; }
      KEMThreeVector& GetNormalizedElectricField() const { return fElectricField;}

    private:
      mutable KPosition fP;
      mutable KEMThreeVector fElectricField;
      KElectrostaticBoundaryIntegrator& fIntegrator;
    };

    mutable ShapeVisitorForPotential fShapeVisitorForPotential;
    mutable ShapeVisitorForElectricField fShapeVisitorForElectricField;
  };
}

#endif /* KELECTROSTATICINTEGRATINGFIELDSOLVER_DEF */
