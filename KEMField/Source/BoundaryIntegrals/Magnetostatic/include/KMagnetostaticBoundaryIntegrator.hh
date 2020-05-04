#ifndef KMAGNETOSTATICBOUNDARYINTEGRATOR_DEF
#define KMAGNETOSTATICBOUNDARYINTEGRATOR_DEF

#include "KMagnetostaticLineSegmentIntegrator.hh"
#include "KMagnetostaticRingIntegrator.hh"
#include "KSurfaceVisitors.hh"

namespace KEMField
{
class KMagnetostaticBoundaryIntegrator : public KMagnetostaticLineSegmentIntegrator, public KMagnetostaticRingIntegrator
{
  public:
    using KMagnetostaticLineSegmentIntegrator::MagneticField;
    using KMagnetostaticLineSegmentIntegrator::VectorPotential;
    using KMagnetostaticRingIntegrator::MagneticField;
    using KMagnetostaticRingIntegrator::VectorPotential;

    typedef KMagnetostaticBasis Basis;
    typedef Basis::ValueType ValueType;
    typedef KBoundaryType<Basis, KDirichletBoundary> DirichletBoundary;
    typedef KBoundaryType<Basis, KNeumannBoundary> NeumannBoundary;

    KMagnetostaticBoundaryIntegrator() : fShapeVisitor(*this) {}
    virtual ~KMagnetostaticBoundaryIntegrator() {}

    ValueType BoundaryIntegral(KSurfacePrimitive* source, KSurfacePrimitive* target, unsigned int i);
    ValueType BoundaryValue(KSurfacePrimitive* surface, unsigned int i);
    ValueType& BasisValue(KSurfacePrimitive* surface, unsigned int i);

  private:
    class ShapeVisitor : public KSelectiveVisitor<KShapeVisitor, KTYPELIST_2(KLineSegment, KRing)>
    {
      public:
        using KSelectiveVisitor<KShapeVisitor, KTYPELIST_2(KLineSegment, KRing)>::Visit;

        ShapeVisitor(KMagnetostaticBoundaryIntegrator& integrator) : fIntegrator(integrator) {}

        void Visit(KLineSegment& l)
        {
            fIntegrator.ComputeBoundaryIntegral(l);
        }
        void Visit(KRing& r)
        {
            fIntegrator.ComputeBoundaryIntegral(r);
        }

      protected:
        KMagnetostaticBoundaryIntegrator& fIntegrator;
    };

    class BoundaryVisitor :
        public KSelectiveVisitor<KBoundaryVisitor, KTYPELIST_2(KDirichletBoundary, KNeumannBoundary)>
    {
      public:
        using KSelectiveVisitor<KBoundaryVisitor, KTYPELIST_2(KDirichletBoundary, KNeumannBoundary)>::Visit;

        BoundaryVisitor() {}

        void Visit(KDirichletBoundary&);
        void Visit(KNeumannBoundary&);

        void SetBoundaryIndex(unsigned int i)
        {
            fBoundaryIndex = i;
        }
        bool IsDirichlet() const
        {
            return fIsDirichlet;
        }
        ValueType Prefactor() const
        {
            return fPrefactor;
        }
        ValueType GetBoundaryValue() const
        {
            return fBoundaryValue;
        }

      protected:
        unsigned int fBoundaryIndex;
        bool fIsDirichlet;
        ValueType fPrefactor;
        ValueType fBoundaryValue;
    };

    class BasisVisitor : public KSelectiveVisitor<KBasisVisitor, KTYPELIST_1(KMagnetostaticBasis)>
    {
      public:
        using KSelectiveVisitor<KBasisVisitor, KTYPELIST_1(KMagnetostaticBasis)>::Visit;

        BasisVisitor() : fBasisValue(NULL) {}

        void Visit(KMagnetostaticBasis&);

        void SetBasisIndex(unsigned int i)
        {
            fBasisIndex = i;
        }
        ValueType& GetBasisValue() const
        {
            return *fBasisValue;
        }

      protected:
        unsigned int fBasisIndex;
        ValueType* fBasisValue;
    };

    template<class SourceShape> void ComputeBoundaryIntegral(SourceShape& source);

    ShapeVisitor fShapeVisitor;
    BoundaryVisitor fBoundaryVisitor;
    BasisVisitor fBasisVisitor;
    KSurfacePrimitive* fTarget;
    ValueType fValue;
};

template<class SourceShape> void KMagnetostaticBoundaryIntegrator::ComputeBoundaryIntegral(SourceShape& source)
{
    // if (fBoundaryVisitor.IsDirichlet())
    // {
    //   KThreeVector A = this->VectorPotential(&source,fTarget->GetShape()->Centroid());
    //   fValue = A.Dot(fTarget->GetShape()->Normal());
    // }
    // else
    // {
    //   KThreeVector field = this->MagneticField(&source,
    // 					       fTarget->GetShape()->Centroid());
    //   fValue = field.Dot(fTarget->GetShape()->Normal());
    //   double dist = (source.Centroid() -
    // 		       fTarget->GetShape()->Centroid()).Magnitude();
    //   if (dist<1.e-12)
    // 	fValue *= fBoundaryVisitor.Prefactor();
    // }
}
}  // namespace KEMField

#endif /* KMAGNETOSTATICBOUNDARYINTEGRATOR_DEF */
