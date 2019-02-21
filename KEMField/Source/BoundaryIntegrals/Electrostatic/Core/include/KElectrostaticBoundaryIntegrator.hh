#ifndef KELECTROSTATICBOUNDARYINTEGRATOR_DEF
#define KELECTROSTATICBOUNDARYINTEGRATOR_DEF

#include "KSurfaceVisitors.hh"
#include "KSurfacePrimitive.hh"
#include "KSmartPointer.hh"
#include "KElectrostaticElementIntegrator.hh"

namespace KEMField
{
  class ElectrostaticSingleThread;

  class KElectrostaticBoundaryIntegrator
  {
  public:
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

    // for field solver template selection, states what kind of boundary integrator this is.
    typedef ElectrostaticSingleThread Kind;

    KElectrostaticBoundaryIntegrator(
            KSmartPointer<KElectrostaticElementIntegrator<KTriangle>> triangleIntegrator,
            KSmartPointer<KElectrostaticElementIntegrator<KRectangle>> rectangleIntegrator,
            KSmartPointer<KElectrostaticElementIntegrator<KLineSegment>> lineSegmentIntegrator,
            KSmartPointer<KElectrostaticElementIntegrator<KConicSection>> conicSectionIntegrator,
            KSmartPointer<KElectrostaticElementIntegrator<KRing>> ringIntegrator
    );

    KElectrostaticBoundaryIntegrator(const KElectrostaticBoundaryIntegrator& integrator);
    KElectrostaticBoundaryIntegrator& operator=(const KElectrostaticBoundaryIntegrator& integrator);
    virtual ~KElectrostaticBoundaryIntegrator() {}

    virtual ValueType BoundaryIntegral(KSurfacePrimitive* source,
				                        unsigned int sourceIndex,
				                        KSurfacePrimitive* target,
				                        unsigned int targetIndex);

    ValueType  BoundaryValue(KSurfacePrimitive* surface,unsigned int);
    ValueType& BasisValue(KSurfacePrimitive* surface,unsigned int);

    //triangle integrator getter and setter and evaluation functions
    KSmartPointer<KElectrostaticElementIntegrator<KTriangle> > GetTriangleIntegrator();

    void SetTriangleIntegrator( KSmartPointer<KElectrostaticElementIntegrator<KTriangle> > triangleIntegrator );

    double Potential( const KTriangle* source, const KPosition& P ) const;
    KThreeVector ElectricField( const KTriangle* source, const KPosition& P) const;
    std::pair<KThreeVector,double> ElectricFieldAndPotential( const KTriangle* source, const KPosition& P) const;

    double Potential( const KTriangleGroup* source, const KPosition& P ) const;
    KThreeVector ElectricField( const KTriangleGroup* source, const KPosition& P ) const;
    std::pair<KThreeVector,double> ElectricFieldAndPotential( const KTriangleGroup* source, const KPosition& P) const;

  private:
    KSmartPointer<KElectrostaticElementIntegrator<KTriangle> > fTriangleIntegrator;

  public:
    //Rectangle integrator getter and setter and evaluation functions
    KSmartPointer<KElectrostaticElementIntegrator<KRectangle> > GetRectangleIntegrator();

    void SetRectangleIntegrator( KSmartPointer<KElectrostaticElementIntegrator<KRectangle> > RectangleIntegrator );

    double Potential( const KRectangle* source, const KPosition& P ) const;
    KThreeVector ElectricField( const KRectangle* source, const KPosition& P) const;
    std::pair<KThreeVector,double> ElectricFieldAndPotential( const KRectangle* source, const KPosition& P) const;

    double Potential( const KRectangleGroup* source, const KPosition& P ) const;
    KThreeVector ElectricField( const KRectangleGroup* source, const KPosition& P ) const;
    std::pair<KThreeVector,double> ElectricFieldAndPotential( const KRectangleGroup* source, const KPosition& P) const;

  private:
    KSmartPointer<KElectrostaticElementIntegrator<KRectangle> > fRectangleIntegrator;

  public:
    //LineSegment integrator getter and setter and evaluation functions
    KSmartPointer<KElectrostaticElementIntegrator<KLineSegment> > GetLineSegmentIntegrator();

    void SetLineSegmentIntegrator( KSmartPointer<KElectrostaticElementIntegrator<KLineSegment> > LineSegmentIntegrator );

    double Potential( const KLineSegment* source, const KPosition& P ) const;
    KThreeVector ElectricField( const KLineSegment* source, const KPosition& P) const;
    std::pair<KThreeVector,double> ElectricFieldAndPotential( const KLineSegment* source, const KPosition& P) const;

    double Potential( const KLineSegmentGroup* source, const KPosition& P ) const;
    KThreeVector ElectricField( const KLineSegmentGroup* source, const KPosition& P ) const;
    std::pair<KThreeVector,double> ElectricFieldAndPotential( const KLineSegmentGroup* source, const KPosition& P) const;

  private:
    KSmartPointer<KElectrostaticElementIntegrator<KLineSegment> > fLineSegmentIntegrator;

  public:
    //ConicSection integrator getter and setter and evaluation functions
    KSmartPointer<KElectrostaticElementIntegrator<KConicSection> > GetConicSectionIntegrator();

    void SetConicSectionIntegrator( KSmartPointer<KElectrostaticElementIntegrator<KConicSection> > ConicSectionIntegrator );

    double Potential( const KConicSection* source, const KPosition& P ) const;
    KThreeVector ElectricField( const KConicSection* source, const KPosition& P) const;
    std::pair<KThreeVector,double> ElectricFieldAndPotential( const KConicSection* source, const KPosition& P) const;

    double Potential( const KConicSectionGroup* source, const KPosition& P ) const;
    KThreeVector ElectricField( const KConicSectionGroup* source, const KPosition& P ) const;
    std::pair<KThreeVector,double> ElectricFieldAndPotential( const KConicSectionGroup* source, const KPosition& P) const;

  private:
    KSmartPointer<KElectrostaticElementIntegrator<KConicSection> > fConicSectionIntegrator;

  public:
    //Ring integrator getter and setter and evaluation functions
    KSmartPointer<KElectrostaticElementIntegrator<KRing> > GetRingIntegrator();

    void SetRingIntegrator( KSmartPointer<KElectrostaticElementIntegrator<KRing> > RingIntegrator );

    double Potential( const KRing* source, const KPosition& P ) const;
    KThreeVector ElectricField( const KRing* source, const KPosition& P) const;
    std::pair<KThreeVector,double> ElectricFieldAndPotential( const KRing* source, const KPosition& P) const;

    double Potential( const KRingGroup* source, const KPosition& P ) const;
    KThreeVector ElectricField( const KRingGroup* source, const KPosition& P ) const;
    std::pair<KThreeVector,double> ElectricFieldAndPotential( const KRingGroup* source, const KPosition& P) const;

  private:
    KSmartPointer<KElectrostaticElementIntegrator<KRing> > fRingIntegrator;

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
      double dist = (source.Centroid() - fTarget->GetShape()->Centroid()).Magnitude();

      if( dist>=1.e-12 ) {
          KThreeVector field = this->ElectricField(&source,fTarget->GetShape()->Centroid());
          fValue = field.Dot(fTarget->GetShape()->Normal());
      } else {
          // For planar Neumann elements (here: triangles and rectangles) the following formula
          // is valid and incorporates already the electric field 1./(2.*Eps0).
          // In case of conical (axialsymmetric) Neumann elements this formula has to be modified.
          // Ferenc Glueck and Daniel Hilk, March 27th 2018
          fValue = fBoundaryVisitor.Prefactor()/(2.*KEMConstants::Eps0);
      }
    }
  }
}

#endif /* KELECTROSTATICBOUNDARYINTEGRATOR_DEF */
