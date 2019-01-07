#include "KElectrostaticBoundaryIntegrator.hh"

#include "KElectrostaticAnalyticTriangleIntegrator.hh"
#include "KElectrostaticAnalyticRectangleIntegrator.hh"
#include "KElectrostaticAnalyticLineSegmentIntegrator.hh"
#include "KElectrostaticAnalyticConicSectionIntegrator.hh"
#include "KElectrostaticAnalyticRingIntegrator.hh"




namespace KEMField
{
KElectrostaticBoundaryIntegrator::KElectrostaticBoundaryIntegrator(
        KSmartPointer<KElectrostaticElementIntegrator<KTriangle>> triangleIntegrator,
        KSmartPointer<KElectrostaticElementIntegrator<KRectangle>> rectangleIntegrator,
        KSmartPointer<KElectrostaticElementIntegrator<KLineSegment>> lineSegmentIntegrator,
        KSmartPointer<KElectrostaticElementIntegrator<KConicSection>> conicSectionIntegrator,
        KSmartPointer<KElectrostaticElementIntegrator<KRing>> ringIntegrator
        ) :
                  fTriangleIntegrator(triangleIntegrator),
                  fRectangleIntegrator(rectangleIntegrator),
                  fLineSegmentIntegrator(lineSegmentIntegrator),
                  fConicSectionIntegrator(conicSectionIntegrator),
                  fRingIntegrator(ringIntegrator),
                  fShapeVisitor(*this)

{
}


KEMField::KElectrostaticBoundaryIntegrator::KElectrostaticBoundaryIntegrator(
		const KElectrostaticBoundaryIntegrator& integrator) :
				KElectrostaticBoundaryIntegrator(
						integrator.fTriangleIntegrator,
						integrator.fRectangleIntegrator,
						integrator.fLineSegmentIntegrator,
						integrator.fConicSectionIntegrator,
						integrator.fRingIntegrator)
{
}

KElectrostaticBoundaryIntegrator& KEMField::KElectrostaticBoundaryIntegrator::operator =(
		const KElectrostaticBoundaryIntegrator& integrator)
{
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
    fPrefactor = ((1. + static_cast<NeumannBoundary&>(boundary).GetNormalBoundaryFlux())/(1. - static_cast<NeumannBoundary&>(boundary).GetNormalBoundaryFlux()));
    fBoundaryValue = 0.;
}

void KElectrostaticBoundaryIntegrator::BasisVisitor::Visit(KElectrostaticBasis& basis)
{
    fBasisValue = &(basis.GetSolution());
}

KElectrostaticBasis::ValueType KElectrostaticBoundaryIntegrator::BoundaryIntegral(KSurfacePrimitive* source,unsigned int,KSurfacePrimitive* target,unsigned int)
{
    fTarget = target;
    target->Accept(fBoundaryVisitor);
    source->Accept(fShapeVisitor);
    return fValue;
}

KElectrostaticBasis::ValueType KElectrostaticBoundaryIntegrator::BoundaryValue(KSurfacePrimitive* surface,unsigned int)
{
    surface->Accept(fBoundaryVisitor);
    return fBoundaryVisitor.GetBoundaryValue();
}

KElectrostaticBasis::ValueType& KElectrostaticBoundaryIntegrator::BasisValue(KSurfacePrimitive* surface,unsigned int)
{
    surface->Accept(fBasisVisitor);
    return fBasisVisitor.GetBasisValue();
}

KSmartPointer<KElectrostaticElementIntegrator<KTriangle> >
KElectrostaticBoundaryIntegrator::GetTriangleIntegrator()
{
    return fTriangleIntegrator;
}

void KElectrostaticBoundaryIntegrator::SetTriangleIntegrator(
        KSmartPointer<KElectrostaticElementIntegrator<KTriangle> > triangleIntegrator )
{
    fTriangleIntegrator = triangleIntegrator;
}

double KElectrostaticBoundaryIntegrator::Potential(
        const KTriangle* source, const KPosition& P) const
{
    return fTriangleIntegrator->Potential(source,P);
}

KThreeVector KElectrostaticBoundaryIntegrator::ElectricField(
        const KTriangle* source, const KPosition& P) const
{
    return fTriangleIntegrator->ElectricField(source,P);
}

std::pair<KThreeVector, double> KElectrostaticBoundaryIntegrator::ElectricFieldAndPotential(
        const KTriangle* source, const KPosition& P) const
{
    return fTriangleIntegrator->ElectricFieldAndPotential(source,P);
}

double KElectrostaticBoundaryIntegrator::Potential(
        const KTriangleGroup* source, const KPosition& P) const
{
    return fTriangleIntegrator->Potential(source,P);
}

KThreeVector KElectrostaticBoundaryIntegrator::ElectricField(
        const KTriangleGroup* source, const KPosition& P) const
{
    return fTriangleIntegrator->ElectricField(source,P);
}

std::pair<KThreeVector, double> KElectrostaticBoundaryIntegrator::ElectricFieldAndPotential(
        const KTriangleGroup* source, const KPosition& P) const
{
    return fTriangleIntegrator->ElectricFieldAndPotential(source,P);
}

KSmartPointer<KElectrostaticElementIntegrator<KRectangle> >
KElectrostaticBoundaryIntegrator::GetRectangleIntegrator()
{
    return fRectangleIntegrator;
}

void KElectrostaticBoundaryIntegrator::SetRectangleIntegrator(
        KSmartPointer<KElectrostaticElementIntegrator<KRectangle> > RectangleIntegrator )
{
    fRectangleIntegrator = RectangleIntegrator;
}

double KElectrostaticBoundaryIntegrator::Potential(
        const KRectangle* source, const KPosition& P) const
{
    return fRectangleIntegrator->Potential(source,P);
}

KThreeVector KElectrostaticBoundaryIntegrator::ElectricField(
        const KRectangle* source, const KPosition& P) const
{
    return fRectangleIntegrator->ElectricField(source,P);
}

std::pair<KThreeVector, double> KElectrostaticBoundaryIntegrator::ElectricFieldAndPotential(
        const KRectangle* source, const KPosition& P) const
{
    return fRectangleIntegrator->ElectricFieldAndPotential(source,P);
}

double KElectrostaticBoundaryIntegrator::Potential(
        const KRectangleGroup* source, const KPosition& P) const
{
    return fRectangleIntegrator->Potential(source,P);
}

KThreeVector KElectrostaticBoundaryIntegrator::ElectricField(
        const KRectangleGroup* source, const KPosition& P) const
{
    return fRectangleIntegrator->ElectricField(source,P);
}

std::pair<KThreeVector, double> KElectrostaticBoundaryIntegrator::ElectricFieldAndPotential(
        const KRectangleGroup* source, const KPosition& P) const
{
    return fRectangleIntegrator->ElectricFieldAndPotential(source,P);
}

KSmartPointer<KElectrostaticElementIntegrator<KLineSegment> >
KElectrostaticBoundaryIntegrator::GetLineSegmentIntegrator()
{
    return fLineSegmentIntegrator;
}

void KElectrostaticBoundaryIntegrator::SetLineSegmentIntegrator(
        KSmartPointer<KElectrostaticElementIntegrator<KLineSegment> > LineSegmentIntegrator )
{
    fLineSegmentIntegrator = LineSegmentIntegrator;
}

double KElectrostaticBoundaryIntegrator::Potential(
        const KLineSegment* source, const KPosition& P) const
{
    return fLineSegmentIntegrator->Potential(source,P);
}

KThreeVector KElectrostaticBoundaryIntegrator::ElectricField(
        const KLineSegment* source, const KPosition& P) const
{
    return fLineSegmentIntegrator->ElectricField(source,P);
}

std::pair<KThreeVector, double> KElectrostaticBoundaryIntegrator::ElectricFieldAndPotential(
        const KLineSegment* source, const KPosition& P) const
{
    return fLineSegmentIntegrator->ElectricFieldAndPotential(source,P);
}

double KElectrostaticBoundaryIntegrator::Potential(
        const KLineSegmentGroup* source, const KPosition& P) const
{
    return fLineSegmentIntegrator->Potential(source,P);
}

KThreeVector KElectrostaticBoundaryIntegrator::ElectricField(
        const KLineSegmentGroup* source, const KPosition& P) const
{
    return fLineSegmentIntegrator->ElectricField(source,P);
}

std::pair<KThreeVector, double> KElectrostaticBoundaryIntegrator::ElectricFieldAndPotential(
        const KLineSegmentGroup* source, const KPosition& P) const
{
    return fLineSegmentIntegrator->ElectricFieldAndPotential(source,P);
}

KSmartPointer<KElectrostaticElementIntegrator<KConicSection> >
KElectrostaticBoundaryIntegrator::GetConicSectionIntegrator()
{
    return fConicSectionIntegrator;
}

void KElectrostaticBoundaryIntegrator::SetConicSectionIntegrator(
        KSmartPointer<KElectrostaticElementIntegrator<KConicSection> > ConicSectionIntegrator )
{
    fConicSectionIntegrator = ConicSectionIntegrator;
}

double KElectrostaticBoundaryIntegrator::Potential(
        const KConicSection* source, const KPosition& P) const
{
    return fConicSectionIntegrator->Potential(source,P);
}

KThreeVector KElectrostaticBoundaryIntegrator::ElectricField(
        const KConicSection* source, const KPosition& P) const
{
    return fConicSectionIntegrator->ElectricField(source,P);
}

std::pair<KThreeVector, double> KElectrostaticBoundaryIntegrator::ElectricFieldAndPotential(
        const KConicSection* source, const KPosition& P) const
{
    return fConicSectionIntegrator->ElectricFieldAndPotential(source,P);
}

double KElectrostaticBoundaryIntegrator::Potential(
        const KConicSectionGroup* source, const KPosition& P) const
{
    return fConicSectionIntegrator->Potential(source,P);
}

KThreeVector KElectrostaticBoundaryIntegrator::ElectricField(
        const KConicSectionGroup* source, const KPosition& P) const
{
    return fConicSectionIntegrator->ElectricField(source,P);
}

std::pair<KThreeVector, double> KElectrostaticBoundaryIntegrator::ElectricFieldAndPotential(
        const KConicSectionGroup* source, const KPosition& P) const
{
    return fConicSectionIntegrator->ElectricFieldAndPotential(source,P);
}


KSmartPointer<KElectrostaticElementIntegrator<KRing> >
KElectrostaticBoundaryIntegrator::GetRingIntegrator()
{
    return fRingIntegrator;
}

void KElectrostaticBoundaryIntegrator::SetRingIntegrator(
        KSmartPointer<KElectrostaticElementIntegrator<KRing> > RingIntegrator )
{
    fRingIntegrator = RingIntegrator;
}

double KElectrostaticBoundaryIntegrator::Potential(
        const KRing* source, const KPosition& P) const
{
    return fRingIntegrator->Potential(source,P);
}

KThreeVector KElectrostaticBoundaryIntegrator::ElectricField(
        const KRing* source, const KPosition& P) const
{
    return fRingIntegrator->ElectricField(source,P);
}

std::pair<KThreeVector, double> KElectrostaticBoundaryIntegrator::ElectricFieldAndPotential(
        const KRing* source, const KPosition& P) const
{
    return fRingIntegrator->ElectricFieldAndPotential(source,P);
}

double KElectrostaticBoundaryIntegrator::Potential(
        const KRingGroup* source, const KPosition& P) const
{
    return fRingIntegrator->Potential(source,P);
}

KThreeVector KElectrostaticBoundaryIntegrator::ElectricField(
        const KRingGroup* source, const KPosition& P) const
{
    return fRingIntegrator->ElectricField(source,P);
}

std::pair<KThreeVector, double> KElectrostaticBoundaryIntegrator::ElectricFieldAndPotential(
        const KRingGroup* source, const KPosition& P) const
{
    return fRingIntegrator->ElectricFieldAndPotential(source,P);
}

} /* KEMField */
