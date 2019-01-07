#include <vector>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <iomanip>

#include "KGConicalWireArraySurface.hh"

#include "KGDiscreteRotationalMesher.hh"

#include "KGBEM.hh"
#include "KGBEMConverter.hh"

#include "KSurface.hh"

#include "KSurfaceContainer.hh"

#include "KElectrostaticBoundaryIntegratorFactory.hh"
#include "KBoundaryIntegralMatrix.hh"
#include "KBoundaryIntegralVector.hh"
#include "KBoundaryIntegralSolutionVector.hh"

#include "KRobinHood.hh"

#include "KEMCout.hh"

#ifdef KEMFIELD_USE_VTK
#include "KEMVTKViewer.hh"
#endif

using namespace KEMField;

int main(int argc, char** argv)
{
  int test = 0;
  if (argc > 1)
    test = atoi(argv[1]);

  if (test == 0)
  {
    double a = 1.5;
    double b = 1.3;
    KThreeVector p0(0.,0.,-1.);
    KThreeVector n1(1./sqrt(2.),1./sqrt(2.),0.);
    KThreeVector n2(1./sqrt(2.),-1./sqrt(2.),0.);

    double dirichletValue = 10.2;

    double chargeDensity = 4.8;

    KSurface<KElectrostaticBasis,KDirichletBoundary,KSymmetryGroup<KRectangle> >* tg = new KSurface<KElectrostaticBasis,KDirichletBoundary,KSymmetryGroup<KRectangle> >();

    KRectangle* t = tg->NewElement();

    t->SetA(a);
    t->SetB(b);
    t->SetP0(p0);
    t->SetN1(n1);
    t->SetN2(n2);

    tg->SetBoundaryValue(dirichletValue);
    tg->SetSolution(chargeDensity);

    KEMField::cout<<"Before symmetry: "<<KEMField::endl;
    KEMField::cout<<*tg<<KEMField::endl;

    tg->AddReflectionThroughPlane(KGeoBag::KThreeVector(0.,0.,0.),
				  KGeoBag::KThreeVector(0.,0.,-1.));

    KEMField::cout<<"After symmetry: "<<KEMField::endl;
    KEMField::cout<<*tg<<KEMField::endl;
  }
  else if (test == 1)
  {
#ifdef KEMFIELD_USE_KGEOBAG
    KGeoBag::KGConicalWireArray* wireArray = new KGeoBag::KGConicalWireArray();
    wireArray->SetR1(1.);
    wireArray->SetZ1(0.);
    wireArray->SetR2(1.5);
    wireArray->SetZ2(1.);
    wireArray->SetNWires(100);
    wireArray->SetDiameter(.003);
    wireArray->SetNDisc(30);

    KGeoBag::KGConicalWireArraySurface* wASurface = new KGeoBag::KGConicalWireArraySurface(wireArray);
    KGeoBag::KGSurface* wireArraySurface = new KGeoBag::KGSurface(wASurface);
    wireArraySurface->SetName("conicalWireArray");
    wireArraySurface->MakeExtension<KGeoBag::KGDiscreteRotationalMesh>();
    wireArraySurface->MakeExtension<KGeoBag::KGElectrostaticDirichlet>();
    wireArraySurface->AsExtension<KGeoBag::KGElectrostaticDirichlet>()->SetBoundaryValue(1.);

    // Mesh the elements
    KGeoBag::KGDiscreteRotationalMesher* mesher = new KGeoBag::KGDiscreteRotationalMesher();
    mesher->SetAxialCount(100);
    wireArraySurface->AcceptNode(mesher);

    KSurfaceContainer surfaceContainer;
    KGeoBag::KGBEMDiscreteRotationalMeshConverter geometryConverter(surfaceContainer);
    wireArraySurface->AcceptNode(&geometryConverter);

    KEMField::cout<<"solving for "<<surfaceContainer.size()<<" elements"<<KEMField::endl;

    KElectrostaticBoundaryIntegrator integrator {KEBIFactory::MakeDefault()};
    KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A(surfaceContainer,
								integrator);
    KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x(surfaceContainer,integrator);
    KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b(surfaceContainer,integrator);

    KRobinHood<KElectrostaticBoundaryIntegrator::ValueType> robinHood;
    robinHood.Solve(A,x,b);

#ifdef KEMFIELD_USE_VTK
    KEMVTKViewer viewer(surfaceContainer);
    viewer.GenerateGeometryFile("conicalWireArray.vtp");
    viewer.ViewGeometry();
#endif

    delete wireArray;
#endif
  }

  return 0;
}
