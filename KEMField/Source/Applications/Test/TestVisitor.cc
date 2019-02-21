#include <vector>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <iomanip>

#include "KThreeVector_KEMField.hh"
#include "KEMConstants.hh"
#include "KEMCout.hh"

#include "KSurface.hh"

using namespace KEMField;

int main(int /*argc*/, char** /*argv*/)
{
  double a = 1.5;
  double b = 1.3;
  KThreeVector p0(0.,0.,0.);
  KThreeVector n1(1./sqrt(2.),1./sqrt(2.),0.);
  KThreeVector n2(1./sqrt(2.),-1./sqrt(2.),0.);

  double dirichletValue = 10.2;

  double chargeDensity = 4.8;

  KSurface<KElectrostaticBasis,
  	   KDirichletBoundary,
  	   KTriangle>* t = new KSurface<KElectrostaticBasis,
  					KDirichletBoundary,
  					KTriangle>();


  t->SetA(a);
  t->SetB(b);
  t->SetP0(p0);
  t->SetN1(n1);
  t->SetN2(n2);

  t->SetBoundaryValue(dirichletValue);

  t->SetSolution(chargeDensity);

  
  class Visitor : public KSelectiveVisitor<KShapeVisitor,
					   KTYPELIST_2( KTriangle,KRectangle )>
  {
  public:
    using KSelectiveVisitor<KShapeVisitor,
			    KTYPELIST_2( KTriangle,KRectangle )>::Visit;

    Visitor() {}
    ~Visitor() {}

    void Visit(KTriangle& t) { (void)t;KEMField::cout<<"Triangle!"<<KEMField::endl; }
    void Visit(KRectangle& r) { (void)r;KEMField::cout<<"Rectangle!"<<KEMField::endl; }
  };

  Visitor visitor;

  t->Accept(visitor);

  KSurface<KElectrostaticBasis,
	   KDirichletBoundary,
	   KRectangle>* r = new KSurface<KElectrostaticBasis,
					KDirichletBoundary,
					KRectangle>();

  r->SetA(a);
  r->SetB(b);
  r->SetP0(p0);
  r->SetN1(n1);
  r->SetN2(n2);

  r->SetBoundaryValue(dirichletValue);

  r->SetSolution(chargeDensity);

  r->Accept(visitor);
  // visitor.Visit(r->GetShape());

  KSurfacePrimitive* p = r;

  KEMField::cout<<"From base: "<<KEMField::endl;

  p->Accept(visitor);
  // visitor.Visit(p->GetBasis());

  KSurface<KElectrostaticBasis,
	   KRobinBoundary,
	   KLineSegment>* w = new KSurface<KElectrostaticBasis,
					   KRobinBoundary,
					   KLineSegment>();

  w->SetP0(KThreeVector(0.,1.,0.));
  w->SetP1(KThreeVector(1.,0.,0.));
  w->SetDiameter(1.e-4);
  w->SetNormalBoundaryFlux(3.3);
  w->SetSolution(12.6);

  KEMField::cout<<"Attempting to visit a class that is not in the selective visitor:"<<KEMField::endl;

  w->Accept(visitor);

  delete r;
  delete t;

  return 0;
}
