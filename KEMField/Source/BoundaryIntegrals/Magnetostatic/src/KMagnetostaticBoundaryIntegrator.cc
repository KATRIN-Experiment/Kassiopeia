#include "KMagnetostaticBoundaryIntegrator.hh"

namespace KEMField
{
void KMagnetostaticBoundaryIntegrator::BoundaryVisitor::Visit(KDirichletBoundary& boundary)
{
    // TO DO: get the right values for magnetostatics!
    fIsDirichlet = true;
    fPrefactor = 1.;
    fBoundaryValue = static_cast<DirichletBoundary&>(boundary).GetBoundaryValue(fBoundaryIndex);
}

void KMagnetostaticBoundaryIntegrator::BoundaryVisitor::Visit(KNeumannBoundary& boundary)
{
    // TO DO: get the right values for magnetostatics!
    fIsDirichlet = false;
    fPrefactor = ((1. + static_cast<NeumannBoundary&>(boundary).GetNormalBoundaryFlux(fBoundaryIndex)) /
                  (1. - static_cast<NeumannBoundary&>(boundary).GetNormalBoundaryFlux(fBoundaryIndex)));
    fBoundaryValue = 0.;
}

void KMagnetostaticBoundaryIntegrator::BasisVisitor::Visit(KMagnetostaticBasis& basis)
{
    fBasisValue = &(basis.GetSolution(fBasisIndex));
}

KMagnetostaticBasis::ValueType
KMagnetostaticBoundaryIntegrator::BoundaryIntegral(KSurfacePrimitive* source, KSurfacePrimitive* target, unsigned int i)
{
    fTarget = target;
    fBoundaryVisitor.SetBoundaryIndex(i);
    target->Accept(fBoundaryVisitor);
    source->Accept(fShapeVisitor);
    return fValue;
}

KMagnetostaticBasis::ValueType KMagnetostaticBoundaryIntegrator::BoundaryValue(KSurfacePrimitive* surface,
                                                                               unsigned int i)
{
    fBoundaryVisitor.SetBoundaryIndex(i);
    surface->Accept(fBoundaryVisitor);
    return fBoundaryVisitor.GetBoundaryValue(i);
}

KMagnetostaticBasis::ValueType& KMagnetostaticBoundaryIntegrator::BasisValue(KSurfacePrimitive* surface, unsigned int i)
{
    fBasisVisitor.SetBasisIndex(i);
    surface->Accept(fBasisVisitor);
    return fBasisVisitor.GetBasisValue(i);
}
}  // namespace KEMField
