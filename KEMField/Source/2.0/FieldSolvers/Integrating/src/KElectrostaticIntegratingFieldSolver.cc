#include "KElectrostaticIntegratingFieldSolver.hh"

namespace KEMField
{
  KIntegratingFieldSolver<KElectrostaticBoundaryIntegrator>::
  KIntegratingFieldSolver(const KSurfaceContainer& container,
			  KElectrostaticBoundaryIntegrator& integrator)
    : fContainer(container),
      fIntegrator(integrator),
      fShapeVisitorForPotential(integrator),
      fShapeVisitorForElectricField(integrator)
  {

  }

  double KIntegratingFieldSolver<KElectrostaticBoundaryIntegrator>::Potential(const KPosition& P) const
  {
    // Kahan Sum to mitigate rounding error
    fShapeVisitorForPotential.SetPosition(P);
    double sum = 0.;
    double c = 0.;
    double y = 0.;
    double t = 0.;
    KSurfaceContainer::iterator it;
    for (it=fContainer.begin<Basis>();it!=fContainer.end<Basis>();++it)
    {
      (*it)->Accept(fShapeVisitorForPotential);
      y = fShapeVisitorForPotential.GetNormalizedPotential()*fIntegrator.BasisValue(*it,0) - c;
      t = sum + y;
      c = (t - sum) - y;
      sum = t;
    }
    return sum;
  }

  KEMThreeVector KIntegratingFieldSolver<KElectrostaticBoundaryIntegrator>::ElectricField(const KPosition& P) const
  {
    // Kahan Sum to mitigate rounding error
    fShapeVisitorForElectricField.SetPosition(P);
    KEMThreeVector sum(0.,0.,0.);
    KEMThreeVector c(0.,0.,0.);
    KEMThreeVector y(0.,0.,0.);
    KEMThreeVector t(0.,0.,0.);
    KSurfaceContainer::iterator it;
    for (it=fContainer.begin<Basis>();it!=fContainer.end<Basis>();++it)
    {
      (*it)->Accept(fShapeVisitorForElectricField);
      y = fShapeVisitorForElectricField.GetNormalizedElectricField()*fIntegrator.BasisValue(*it,0) - c;
      t = sum + y;
      c = (t - sum) - y;
      sum = t;
    }
    return sum;
  }

  double KIntegratingFieldSolver<KElectrostaticBoundaryIntegrator>::Potential(const unsigned int* SurfaceIndexSet, unsigned int SetSize, const KPosition& P) const
  {
    // Kahan Sum to mitigate rounding error
    fShapeVisitorForPotential.SetPosition(P);
    double sum = 0.;
    double c = 0.;
    double y = 0.;
    double t = 0.;
    unsigned int id;
    for(unsigned int i=0; i<SetSize; ++i)
    {
      id = SurfaceIndexSet[i];
      fContainer[id]->Accept(fShapeVisitorForPotential);
      y = fShapeVisitorForPotential.GetNormalizedPotential()*fIntegrator.BasisValue(fContainer[id],0) - c;
      t = sum + y;
      c = (t - sum) - y;
      sum = t;
    }
    return sum;
  }


  KEMThreeVector KIntegratingFieldSolver<KElectrostaticBoundaryIntegrator>::ElectricField(const unsigned int* SurfaceIndexSet, unsigned int SetSize, const KPosition& P) const
  {
    // Kahan Sum to mitigate rounding error
    fShapeVisitorForElectricField.SetPosition(P);
    KEMThreeVector sum(0.,0.,0.);
    KEMThreeVector c(0.,0.,0.);
    KEMThreeVector y(0.,0.,0.);
    KEMThreeVector t(0.,0.,0.);
    unsigned int id;
    for(unsigned int i=0; i<SetSize; ++i)
    {
      id = SurfaceIndexSet[i];
      fContainer[id]->Accept(fShapeVisitorForElectricField);
      y = fShapeVisitorForElectricField.GetNormalizedElectricField()*fIntegrator.BasisValue(fContainer[id],0) - c;
      t = sum + y;
      c = (t - sum) - y;
      sum = t;
    }
    return sum;
  }


}
