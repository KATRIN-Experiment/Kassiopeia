#include "KElectrostaticBoundaryIntegrator.hh"

namespace KEMField
{
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
}
