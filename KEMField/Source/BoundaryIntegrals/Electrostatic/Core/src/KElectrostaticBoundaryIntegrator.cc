#include "KElectrostaticBoundaryIntegrator.hh"

#include "KElectrostaticAnalyticConicSectionIntegrator.hh"
#include "KElectrostaticAnalyticLineSegmentIntegrator.hh"
#include "KElectrostaticAnalyticRectangleIntegrator.hh"
#include "KElectrostaticAnalyticRingIntegrator.hh"
#include "KElectrostaticAnalyticTriangleIntegrator.hh"


namespace KEMField
{
KElectrostaticBoundaryIntegrator::KElectrostaticBoundaryIntegrator(
    const KSmartPointer<KElectrostaticElementIntegrator<KTriangle>>& triangleIntegrator,
    const KSmartPointer<KElectrostaticElementIntegrator<KRectangle>>& rectangleIntegrator,
    const KSmartPointer<KElectrostaticElementIntegrator<KLineSegment>>& lineSegmentIntegrator,
    const KSmartPointer<KElectrostaticElementIntegrator<KConicSection>>& conicSectionIntegrator,
    const KSmartPointer<KElectrostaticElementIntegrator<KRing>>& ringIntegrator) :
    fTriangleIntegrator(triangleIntegrator),
    fRectangleIntegrator(rectangleIntegrator),
    fLineSegmentIntegrator(lineSegmentIntegrator),
    fConicSectionIntegrator(conicSectionIntegrator),
    fRingIntegrator(ringIntegrator),
    fShapeVisitor(*this)

{}


KEMField::KElectrostaticBoundaryIntegrator::KElectrostaticBoundaryIntegrator(
    const KElectrostaticBoundaryIntegrator& integrator) :
    KElectrostaticBoundaryIntegrator(integrator.fTriangleIntegrator, integrator.fRectangleIntegrator,
                                     integrator.fLineSegmentIntegrator, integrator.fConicSectionIntegrator,
                                     integrator.fRingIntegrator)
{}

KElectrostaticBoundaryIntegrator&
KEMField::KElectrostaticBoundaryIntegrator::operator=(const KElectrostaticBoundaryIntegrator& integrator)
{
    if (this == &integrator)
        return *this;

    fTriangleIntegrator = integrator.fTriangleIntegrator;
    fRectangleIntegrator = integrator.fRectangleIntegrator;
    fLineSegmentIntegrator = integrator.fLineSegmentIntegrator;
    fConicSectionIntegrator = integrator.fConicSectionIntegrator;
    fRingIntegrator = integrator.fRingIntegrator;
    return *this;
}

void KElectrostaticBoundaryIntegrator::BoundaryVisitor::Visit(KDirichletBoundary& boundary)
{
    fIsDirichlet = true;
    fPrefactor = 1.;
    fBoundaryValue = static_cast<DirichletBoundary&>(boundary).GetBoundaryValue();
}

void KElectrostaticBoundaryIntegrator::BoundaryVisitor::Visit(KNeumannBoundary& boundary)
{
    fIsDirichlet = false;
    fPrefactor = ((1. + static_cast<NeumannBoundary&>(boundary).GetNormalBoundaryFlux()) /
                  (1. - static_cast<NeumannBoundary&>(boundary).GetNormalBoundaryFlux()));
    fBoundaryValue = 0.;
}

void KElectrostaticBoundaryIntegrator::BasisVisitor::Visit(KElectrostaticBasis& basis)
{
    fBasisValue = &(basis.GetSolution());
}

KElectrostaticBasis::ValueType KElectrostaticBoundaryIntegrator::BoundaryIntegral(KSurfacePrimitive* source,
                                                                                  unsigned int /*unused*/,
                                                                                  KSurfacePrimitive* target,
                                                                                  unsigned int /*unused*/)
{
    fTarget = target;
    target->Accept(fBoundaryVisitor);
    source->Accept(fShapeVisitor);
    return fValue;
}

KElectrostaticBasis::ValueType KElectrostaticBoundaryIntegrator::BoundaryValue(KSurfacePrimitive* surface,
                                                                               unsigned int /*unused*/)
{
    surface->Accept(fBoundaryVisitor);
    return fBoundaryVisitor.GetBoundaryValue();
}

KElectrostaticBasis::ValueType& KElectrostaticBoundaryIntegrator::BasisValue(KSurfacePrimitive* surface,
                                                                             unsigned int /*unused*/)
{
    surface->Accept(fBasisVisitor);
    return fBasisVisitor.GetBasisValue();
}

KSmartPointer<KElectrostaticElementIntegrator<KTriangle>> KElectrostaticBoundaryIntegrator::GetTriangleIntegrator()
{
    return fTriangleIntegrator;
}

void KElectrostaticBoundaryIntegrator::SetTriangleIntegrator(
    const KSmartPointer<KElectrostaticElementIntegrator<KTriangle>>& triangleIntegrator)
{
    fTriangleIntegrator = triangleIntegrator;
}

double KElectrostaticBoundaryIntegrator::Potential(const KTriangle* source, const KPosition& P) const
{
    return fTriangleIntegrator->Potential(source, P);
}

KFieldVector KElectrostaticBoundaryIntegrator::ElectricField(const KTriangle* source, const KPosition& P) const
{
    return fTriangleIntegrator->ElectricField(source, P);
}

std::pair<KFieldVector, double> KElectrostaticBoundaryIntegrator::ElectricFieldAndPotential(const KTriangle* source,
                                                                                            const KPosition& P) const
{
    return fTriangleIntegrator->ElectricFieldAndPotential(source, P);
}

double KElectrostaticBoundaryIntegrator::Potential(const KTriangleGroup* source, const KPosition& P) const
{
    return fTriangleIntegrator->Potential(source, P);
}

KFieldVector KElectrostaticBoundaryIntegrator::ElectricField(const KTriangleGroup* source, const KPosition& P) const
{
    return fTriangleIntegrator->ElectricField(source, P);
}

std::pair<KFieldVector, double>
KElectrostaticBoundaryIntegrator::ElectricFieldAndPotential(const KTriangleGroup* source, const KPosition& P) const
{
    return fTriangleIntegrator->ElectricFieldAndPotential(source, P);
}

KSmartPointer<KElectrostaticElementIntegrator<KRectangle>> KElectrostaticBoundaryIntegrator::GetRectangleIntegrator()
{
    return fRectangleIntegrator;
}

void KElectrostaticBoundaryIntegrator::SetRectangleIntegrator(
    const KSmartPointer<KElectrostaticElementIntegrator<KRectangle>>& RectangleIntegrator)
{
    fRectangleIntegrator = RectangleIntegrator;
}

double KElectrostaticBoundaryIntegrator::Potential(const KRectangle* source, const KPosition& P) const
{
    return fRectangleIntegrator->Potential(source, P);
}

KFieldVector KElectrostaticBoundaryIntegrator::ElectricField(const KRectangle* source, const KPosition& P) const
{
    return fRectangleIntegrator->ElectricField(source, P);
}

std::pair<KFieldVector, double> KElectrostaticBoundaryIntegrator::ElectricFieldAndPotential(const KRectangle* source,
                                                                                            const KPosition& P) const
{
    return fRectangleIntegrator->ElectricFieldAndPotential(source, P);
}

double KElectrostaticBoundaryIntegrator::Potential(const KRectangleGroup* source, const KPosition& P) const
{
    return fRectangleIntegrator->Potential(source, P);
}

KFieldVector KElectrostaticBoundaryIntegrator::ElectricField(const KRectangleGroup* source, const KPosition& P) const
{
    return fRectangleIntegrator->ElectricField(source, P);
}

std::pair<KFieldVector, double>
KElectrostaticBoundaryIntegrator::ElectricFieldAndPotential(const KRectangleGroup* source, const KPosition& P) const
{
    return fRectangleIntegrator->ElectricFieldAndPotential(source, P);
}

KSmartPointer<KElectrostaticElementIntegrator<KLineSegment>>
KElectrostaticBoundaryIntegrator::GetLineSegmentIntegrator()
{
    return fLineSegmentIntegrator;
}

void KElectrostaticBoundaryIntegrator::SetLineSegmentIntegrator(
    const KSmartPointer<KElectrostaticElementIntegrator<KLineSegment>>& LineSegmentIntegrator)
{
    fLineSegmentIntegrator = LineSegmentIntegrator;
}

double KElectrostaticBoundaryIntegrator::Potential(const KLineSegment* source, const KPosition& P) const
{
    return fLineSegmentIntegrator->Potential(source, P);
}

KFieldVector KElectrostaticBoundaryIntegrator::ElectricField(const KLineSegment* source, const KPosition& P) const
{
    return fLineSegmentIntegrator->ElectricField(source, P);
}

std::pair<KFieldVector, double> KElectrostaticBoundaryIntegrator::ElectricFieldAndPotential(const KLineSegment* source,
                                                                                            const KPosition& P) const
{
    return fLineSegmentIntegrator->ElectricFieldAndPotential(source, P);
}

double KElectrostaticBoundaryIntegrator::Potential(const KLineSegmentGroup* source, const KPosition& P) const
{
    return fLineSegmentIntegrator->Potential(source, P);
}

KFieldVector KElectrostaticBoundaryIntegrator::ElectricField(const KLineSegmentGroup* source, const KPosition& P) const
{
    return fLineSegmentIntegrator->ElectricField(source, P);
}

std::pair<KFieldVector, double>
KElectrostaticBoundaryIntegrator::ElectricFieldAndPotential(const KLineSegmentGroup* source, const KPosition& P) const
{
    return fLineSegmentIntegrator->ElectricFieldAndPotential(source, P);
}

KSmartPointer<KElectrostaticElementIntegrator<KConicSection>>
KElectrostaticBoundaryIntegrator::GetConicSectionIntegrator()
{
    return fConicSectionIntegrator;
}

void KElectrostaticBoundaryIntegrator::SetConicSectionIntegrator(
    const KSmartPointer<KElectrostaticElementIntegrator<KConicSection>>& ConicSectionIntegrator)
{
    fConicSectionIntegrator = ConicSectionIntegrator;
}

double KElectrostaticBoundaryIntegrator::Potential(const KConicSection* source, const KPosition& P) const
{
    return fConicSectionIntegrator->Potential(source, P);
}

KFieldVector KElectrostaticBoundaryIntegrator::ElectricField(const KConicSection* source, const KPosition& P) const
{
    return fConicSectionIntegrator->ElectricField(source, P);
}

std::pair<KFieldVector, double> KElectrostaticBoundaryIntegrator::ElectricFieldAndPotential(const KConicSection* source,
                                                                                            const KPosition& P) const
{
    return fConicSectionIntegrator->ElectricFieldAndPotential(source, P);
}

double KElectrostaticBoundaryIntegrator::Potential(const KConicSectionGroup* source, const KPosition& P) const
{
    return fConicSectionIntegrator->Potential(source, P);
}

KFieldVector KElectrostaticBoundaryIntegrator::ElectricField(const KConicSectionGroup* source, const KPosition& P) const
{
    return fConicSectionIntegrator->ElectricField(source, P);
}

std::pair<KFieldVector, double>
KElectrostaticBoundaryIntegrator::ElectricFieldAndPotential(const KConicSectionGroup* source, const KPosition& P) const
{
    return fConicSectionIntegrator->ElectricFieldAndPotential(source, P);
}


KSmartPointer<KElectrostaticElementIntegrator<KRing>> KElectrostaticBoundaryIntegrator::GetRingIntegrator()
{
    return fRingIntegrator;
}

void KElectrostaticBoundaryIntegrator::SetRingIntegrator(
    const KSmartPointer<KElectrostaticElementIntegrator<KRing>>& RingIntegrator)
{
    fRingIntegrator = RingIntegrator;
}

double KElectrostaticBoundaryIntegrator::Potential(const KRing* source, const KPosition& P) const
{
    return fRingIntegrator->Potential(source, P);
}

KFieldVector KElectrostaticBoundaryIntegrator::ElectricField(const KRing* source, const KPosition& P) const
{
    return fRingIntegrator->ElectricField(source, P);
}

std::pair<KFieldVector, double> KElectrostaticBoundaryIntegrator::ElectricFieldAndPotential(const KRing* source,
                                                                                            const KPosition& P) const
{
    return fRingIntegrator->ElectricFieldAndPotential(source, P);
}

double KElectrostaticBoundaryIntegrator::Potential(const KRingGroup* source, const KPosition& P) const
{
    return fRingIntegrator->Potential(source, P);
}

KFieldVector KElectrostaticBoundaryIntegrator::ElectricField(const KRingGroup* source, const KPosition& P) const
{
    return fRingIntegrator->ElectricField(source, P);
}

std::pair<KFieldVector, double> KElectrostaticBoundaryIntegrator::ElectricFieldAndPotential(const KRingGroup* source,
                                                                                            const KPosition& P) const
{
    return fRingIntegrator->ElectricFieldAndPotential(source, P);
}

}  // namespace KEMField
