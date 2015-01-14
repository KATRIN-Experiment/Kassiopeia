#ifndef KELECTROSTATICBOUNDARYINTEGRATOR_DEF
#define KELECTROSTATICBOUNDARYINTEGRATOR_DEF

#include "KSurfaceVisitors.hh"

#include "KElectrostaticTriangleIntegrator.hh"
#include "KElectrostaticRectangleIntegrator.hh"
#include "KElectrostaticLineSegmentIntegrator.hh"
#include "KElectrostaticConicSectionIntegrator.hh"
#include "KElectrostaticRingIntegrator.hh"

namespace KEMField
{
  class KElectrostaticBoundaryIntegrator :
    public KElectrostaticTriangleIntegrator,
    public KElectrostaticRectangleIntegrator,
    public KElectrostaticLineSegmentIntegrator,
    public KElectrostaticConicSectionIntegrator,
    public KElectrostaticRingIntegrator
  {
  public:
    using KElectrostaticTriangleIntegrator::Potential;
    using KElectrostaticRectangleIntegrator::Potential;
    using KElectrostaticLineSegmentIntegrator::Potential;
    using KElectrostaticConicSectionIntegrator::Potential;
    using KElectrostaticRingIntegrator::Potential;
    using KElectrostaticTriangleIntegrator::ElectricField;
    using KElectrostaticRectangleIntegrator::ElectricField;
    using KElectrostaticLineSegmentIntegrator::ElectricField;
    using KElectrostaticConicSectionIntegrator::ElectricField;
    using KElectrostaticRingIntegrator::ElectricField;

    typedef KElectrostaticBasis Basis;
    typedef Basis::ValueType ValueType;
    typedef KBoundaryType<Basis,KDirichletBoundary> DirichletBoundary;
    typedef KBoundaryType<Basis,KNeumannBoundary> NeumannBoundary;
    typedef KTYPELIST_1(KElectrostaticBasis) AcceptedBasis;
    typedef KTYPELIST_2(KDirichletBoundary,
			KNeumannBoundary) AcceptedBoundaries;
    typedef KTYPELIST_10(KTriangle,
			 KRectangle,
			 KLineSegment,
			 KConicSection,
			 KRing,
			 KRectangleGroup,
			 KLineSegmentGroup,
			 KTriangleGroup,
			 KConicSectionGroup,
			 KRingGroup) AcceptedShapes;

    KElectrostaticBoundaryIntegrator() : fShapeVisitor(*this) {}
    virtual ~KElectrostaticBoundaryIntegrator() {}

    virtual ValueType BoundaryIntegral(KSurfacePrimitive* source,
				                        unsigned int sourceIndex,
				                        KSurfacePrimitive* target,
				                        unsigned int targetIndex);

    ValueType  BoundaryValue(KSurfacePrimitive* surface,unsigned int);
    ValueType& BasisValue(KSurfacePrimitive* surface,unsigned int);

  protected:

    class ShapeVisitor : public KSelectiveVisitor<KShapeVisitor,AcceptedShapes>
    {
    public:
      using KSelectiveVisitor<KShapeVisitor,AcceptedShapes>::Visit;

      ShapeVisitor(KElectrostaticBoundaryIntegrator& integrator)
      : fIntegrator(integrator) {}

      void Visit(KTriangle& t) { fIntegrator.ComputeBoundaryIntegral(t); }
      void Visit(KRectangle& r) { fIntegrator.ComputeBoundaryIntegral(r); }
      void Visit(KLineSegment& l) { fIntegrator.ComputeBoundaryIntegral(l); }
      void Visit(KConicSection& c) { fIntegrator.ComputeBoundaryIntegral(c); }
      void Visit(KRing& r) { fIntegrator.ComputeBoundaryIntegral(r); }
      void Visit(KTriangleGroup& t) { fIntegrator.ComputeBoundaryIntegral(t); }
      void Visit(KRectangleGroup& r) { fIntegrator.ComputeBoundaryIntegral(r); }
      void Visit(KLineSegmentGroup& l) { fIntegrator.ComputeBoundaryIntegral(l);}
      void Visit(KConicSectionGroup& c) { fIntegrator.ComputeBoundaryIntegral(c); }
      void Visit(KRingGroup& r) { fIntegrator.ComputeBoundaryIntegral(r); }

    protected:
      KElectrostaticBoundaryIntegrator& fIntegrator;
    };

    class BoundaryVisitor :
      public KSelectiveVisitor<KBoundaryVisitor,AcceptedBoundaries>
    {
    public:
      using KSelectiveVisitor<KBoundaryVisitor,AcceptedBoundaries>::Visit;

      BoundaryVisitor() {}

      void Visit(KDirichletBoundary&);
      void Visit(KNeumannBoundary&);

      bool IsDirichlet() const { return fIsDirichlet; }
      ValueType Prefactor() const { return fPrefactor; }
      ValueType GetBoundaryValue() const { return fBoundaryValue; }

    protected:

      bool fIsDirichlet;
      ValueType fPrefactor;
      ValueType fBoundaryValue;
    };

    class BasisVisitor :
      public KSelectiveVisitor<KBasisVisitor,AcceptedBasis>
    {
    public:
      using KSelectiveVisitor<KBasisVisitor,AcceptedBasis>::Visit;

      BasisVisitor() : fBasisValue(NULL) {}

      void Visit(KElectrostaticBasis&);

      ValueType& GetBasisValue() const { return *fBasisValue; }

    protected:

      ValueType* fBasisValue;
    };

    template <class SourceShape>
    void ComputeBoundaryIntegral(SourceShape& source);

    ShapeVisitor fShapeVisitor;
    BoundaryVisitor fBoundaryVisitor;
    BasisVisitor fBasisVisitor;
    KSurfacePrimitive* fTarget;
    ValueType fValue;
  };

  template <class SourceShape>
  void KElectrostaticBoundaryIntegrator::ComputeBoundaryIntegral(SourceShape& source)
  {
    if (fBoundaryVisitor.IsDirichlet())
    {
      fValue = this->Potential(&source,fTarget->GetShape()->Centroid());
    }
    else
    {
      KEMThreeVector field = this->ElectricField(&source,
					       fTarget->GetShape()->Centroid());
      fValue = field.Dot(fTarget->GetShape()->Normal());
      double dist = (source.Centroid() -
		       fTarget->GetShape()->Centroid()).Magnitude();
      if (dist<1.e-12)
	fValue *= fBoundaryVisitor.Prefactor();
    }
  }
}

#endif /* KELECTROSTATICBOUNDARYINTEGRATOR_DEF */
