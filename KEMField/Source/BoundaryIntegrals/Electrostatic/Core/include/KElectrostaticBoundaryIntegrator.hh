#ifndef KELECTROSTATICBOUNDARYINTEGRATOR_DEF
#define KELECTROSTATICBOUNDARYINTEGRATOR_DEF

#include "KElectrostaticElementIntegrator.hh"
#include "KSmartPointer.hh"
#include "KSurfacePrimitive.hh"
#include "KSurfaceVisitors.hh"

namespace KEMField
{
class ElectrostaticSingleThread;

class KElectrostaticBoundaryIntegrator
{
  public:
    using Basis = KElectrostaticBasis;
    using ValueType = Basis::ValueType;
    using DirichletBoundary = KBoundaryType<Basis, KDirichletBoundary>;
    using NeumannBoundary = KBoundaryType<Basis, KNeumannBoundary>;
    using AcceptedBasis = KEMField::KTypelist<KElectrostaticBasis, KEMField::KNullType>;
    using AcceptedBoundaries =
        KEMField::KTypelist<KDirichletBoundary, KEMField::KTypelist<KNeumannBoundary, KEMField::KNullType>>;
    using AcceptedShapes = KEMField::KTypelist<
        KTriangle,
        KEMField::KTypelist<
            KRectangle,
            KEMField::KTypelist<
                KLineSegment,
                KEMField::KTypelist<
                    KConicSection,
                    KEMField::KTypelist<
                        KRing,
                        KEMField::KTypelist<
                            KRectangleGroup,
                            KEMField::KTypelist<
                                KLineSegmentGroup,
                                KEMField::KTypelist<
                                    KTriangleGroup,
                                    KEMField::KTypelist<KConicSectionGroup,
                                                        KEMField::KTypelist<KRingGroup, KEMField::KNullType>>>>>>>>>>;

    // for field solver template selection, states what kind of boundary integrator this is.
    using Kind = ElectrostaticSingleThread;

    KElectrostaticBoundaryIntegrator(
        const KSmartPointer<KElectrostaticElementIntegrator<KTriangle>>& triangleIntegrator,
        const KSmartPointer<KElectrostaticElementIntegrator<KRectangle>>& rectangleIntegrator,
        const KSmartPointer<KElectrostaticElementIntegrator<KLineSegment>>& lineSegmentIntegrator,
        const KSmartPointer<KElectrostaticElementIntegrator<KConicSection>>& conicSectionIntegrator,
        const KSmartPointer<KElectrostaticElementIntegrator<KRing>>& ringIntegrator);

    KElectrostaticBoundaryIntegrator(const KElectrostaticBoundaryIntegrator& integrator);
    KElectrostaticBoundaryIntegrator& operator=(const KElectrostaticBoundaryIntegrator& integrator);
    virtual ~KElectrostaticBoundaryIntegrator() = default;

    virtual ValueType BoundaryIntegral(KSurfacePrimitive* source, unsigned int sourceIndex, KSurfacePrimitive* target,
                                       unsigned int targetIndex);

    ValueType BoundaryValue(KSurfacePrimitive* surface, unsigned int);
    ValueType& BasisValue(KSurfacePrimitive* surface, unsigned int);

    //triangle integrator getter and setter and evaluation functions
    KSmartPointer<KElectrostaticElementIntegrator<KTriangle>> GetTriangleIntegrator();

    void SetTriangleIntegrator(const KSmartPointer<KElectrostaticElementIntegrator<KTriangle>>& triangleIntegrator);

    double Potential(const KTriangle* source, const KPosition& P) const;
    KFieldVector ElectricField(const KTriangle* source, const KPosition& P) const;
    std::pair<KFieldVector, double> ElectricFieldAndPotential(const KTriangle* source, const KPosition& P) const;

    double Potential(const KTriangleGroup* source, const KPosition& P) const;
    KFieldVector ElectricField(const KTriangleGroup* source, const KPosition& P) const;
    std::pair<KFieldVector, double> ElectricFieldAndPotential(const KTriangleGroup* source, const KPosition& P) const;

  private:
    KSmartPointer<KElectrostaticElementIntegrator<KTriangle>> fTriangleIntegrator;

  public:
    //Rectangle integrator getter and setter and evaluation functions
    KSmartPointer<KElectrostaticElementIntegrator<KRectangle>> GetRectangleIntegrator();

    void SetRectangleIntegrator(const KSmartPointer<KElectrostaticElementIntegrator<KRectangle>>& RectangleIntegrator);

    double Potential(const KRectangle* source, const KPosition& P) const;
    KFieldVector ElectricField(const KRectangle* source, const KPosition& P) const;
    std::pair<KFieldVector, double> ElectricFieldAndPotential(const KRectangle* source, const KPosition& P) const;

    double Potential(const KRectangleGroup* source, const KPosition& P) const;
    KFieldVector ElectricField(const KRectangleGroup* source, const KPosition& P) const;
    std::pair<KFieldVector, double> ElectricFieldAndPotential(const KRectangleGroup* source, const KPosition& P) const;

  private:
    KSmartPointer<KElectrostaticElementIntegrator<KRectangle>> fRectangleIntegrator;

  public:
    //LineSegment integrator getter and setter and evaluation functions
    KSmartPointer<KElectrostaticElementIntegrator<KLineSegment>> GetLineSegmentIntegrator();

    void
    SetLineSegmentIntegrator(const KSmartPointer<KElectrostaticElementIntegrator<KLineSegment>>& LineSegmentIntegrator);

    double Potential(const KLineSegment* source, const KPosition& P) const;
    KFieldVector ElectricField(const KLineSegment* source, const KPosition& P) const;
    std::pair<KFieldVector, double> ElectricFieldAndPotential(const KLineSegment* source, const KPosition& P) const;

    double Potential(const KLineSegmentGroup* source, const KPosition& P) const;
    KFieldVector ElectricField(const KLineSegmentGroup* source, const KPosition& P) const;
    std::pair<KFieldVector, double> ElectricFieldAndPotential(const KLineSegmentGroup* source,
                                                              const KPosition& P) const;

  private:
    KSmartPointer<KElectrostaticElementIntegrator<KLineSegment>> fLineSegmentIntegrator;

  public:
    //ConicSection integrator getter and setter and evaluation functions
    KSmartPointer<KElectrostaticElementIntegrator<KConicSection>> GetConicSectionIntegrator();

    void SetConicSectionIntegrator(
        const KSmartPointer<KElectrostaticElementIntegrator<KConicSection>>& ConicSectionIntegrator);

    double Potential(const KConicSection* source, const KPosition& P) const;
    KFieldVector ElectricField(const KConicSection* source, const KPosition& P) const;
    std::pair<KFieldVector, double> ElectricFieldAndPotential(const KConicSection* source, const KPosition& P) const;

    double Potential(const KConicSectionGroup* source, const KPosition& P) const;
    KFieldVector ElectricField(const KConicSectionGroup* source, const KPosition& P) const;
    std::pair<KFieldVector, double> ElectricFieldAndPotential(const KConicSectionGroup* source,
                                                              const KPosition& P) const;

  private:
    KSmartPointer<KElectrostaticElementIntegrator<KConicSection>> fConicSectionIntegrator;

  public:
    //Ring integrator getter and setter and evaluation functions
    KSmartPointer<KElectrostaticElementIntegrator<KRing>> GetRingIntegrator();

    void SetRingIntegrator(const KSmartPointer<KElectrostaticElementIntegrator<KRing>>& ringIntegrator);

    double Potential(const KRing* source, const KPosition& P) const;
    KFieldVector ElectricField(const KRing* source, const KPosition& P) const;
    std::pair<KFieldVector, double> ElectricFieldAndPotential(const KRing* source, const KPosition& P) const;

    double Potential(const KRingGroup* source, const KPosition& P) const;
    KFieldVector ElectricField(const KRingGroup* source, const KPosition& P) const;
    std::pair<KFieldVector, double> ElectricFieldAndPotential(const KRingGroup* source, const KPosition& P) const;

  private:
    KSmartPointer<KElectrostaticElementIntegrator<KRing>> fRingIntegrator;

  protected:
    class ShapeVisitor : public KSelectiveVisitor<KShapeVisitor, AcceptedShapes>
    {
      public:
        using KSelectiveVisitor<KShapeVisitor, AcceptedShapes>::Visit;

        ShapeVisitor(KElectrostaticBoundaryIntegrator& integrator) : fIntegrator(integrator) {}

        void Visit(KTriangle& t) override
        {
            fIntegrator.ComputeBoundaryIntegral(t);
        }
        void Visit(KRectangle& r) override
        {
            fIntegrator.ComputeBoundaryIntegral(r);
        }
        void Visit(KLineSegment& l) override
        {
            fIntegrator.ComputeBoundaryIntegral(l);
        }
        void Visit(KConicSection& c) override
        {
            fIntegrator.ComputeBoundaryIntegral(c);
        }
        void Visit(KRing& r) override
        {
            fIntegrator.ComputeBoundaryIntegral(r);
        }
        void Visit(KTriangleGroup& t) override
        {
            fIntegrator.ComputeBoundaryIntegral(t);
        }
        void Visit(KRectangleGroup& r) override
        {
            fIntegrator.ComputeBoundaryIntegral(r);
        }
        void Visit(KLineSegmentGroup& l) override
        {
            fIntegrator.ComputeBoundaryIntegral(l);
        }
        void Visit(KConicSectionGroup& c) override
        {
            fIntegrator.ComputeBoundaryIntegral(c);
        }
        void Visit(KRingGroup& r) override
        {
            fIntegrator.ComputeBoundaryIntegral(r);
        }

      protected:
        KElectrostaticBoundaryIntegrator& fIntegrator;
    };

    class BoundaryVisitor : public KSelectiveVisitor<KBoundaryVisitor, AcceptedBoundaries>
    {
      public:
        using KSelectiveVisitor<KBoundaryVisitor, AcceptedBoundaries>::Visit;

        BoundaryVisitor() = default;

        void Visit(KDirichletBoundary&) override;
        void Visit(KNeumannBoundary&) override;

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
        bool fIsDirichlet;
        ValueType fPrefactor;
        ValueType fBoundaryValue;
    };

    class BasisVisitor : public KSelectiveVisitor<KBasisVisitor, AcceptedBasis>
    {
      public:
        using KSelectiveVisitor<KBasisVisitor, AcceptedBasis>::Visit;

        BasisVisitor() : fBasisValue(nullptr) {}

        void Visit(KElectrostaticBasis&) override;

        ValueType& GetBasisValue() const
        {
            return *fBasisValue;
        }

      protected:
        ValueType* fBasisValue;
    };

    template<class SourceShape> void ComputeBoundaryIntegral(SourceShape& source);

    ShapeVisitor fShapeVisitor;
    BoundaryVisitor fBoundaryVisitor;
    BasisVisitor fBasisVisitor;
    KSurfacePrimitive* fTarget;
    ValueType fValue;
};

template<class SourceShape> void KElectrostaticBoundaryIntegrator::ComputeBoundaryIntegral(SourceShape& source)
{
    if (fBoundaryVisitor.IsDirichlet()) {
        fValue = this->Potential(&source, fTarget->GetShape()->Centroid());
    }
    else {
        double dist = (source.Centroid() - fTarget->GetShape()->Centroid()).Magnitude();

        if (dist >= 1.e-12) {
            KFieldVector field = this->ElectricField(&source, fTarget->GetShape()->Centroid());
            fValue = field.Dot(fTarget->GetShape()->Normal());
        }
        else {
            // For planar Neumann elements (here: triangles and rectangles) the following formula
            // is valid and incorporates already the electric field 1./(2.*Eps0).
            // In case of conical (axialsymmetric) Neumann elements this formula has to be modified.
            // Ferenc Glueck and Daniel Hilk, March 27th 2018
            fValue = fBoundaryVisitor.Prefactor() / (2. * KEMConstants::Eps0);
        }
    }
}
}  // namespace KEMField

#endif /* KELECTROSTATICBOUNDARYINTEGRATOR_DEF */
