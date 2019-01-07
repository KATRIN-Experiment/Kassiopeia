#include <vector>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <iomanip>

#include <gsl/gsl_rng.h>

#include "KSurfaceTypes.hh"
#include "KSurface.hh"

#include "KElectrostaticBoundaryIntegratorFactory.hh"

#include "KSurfaceContainer.hh"
#include "KBoundaryIntegralMatrix.hh"

#ifdef KEMFIELD_USE_OPENCL
#include "KOpenCLSurfaceContainer.hh"
#include "KOpenCLElectrostaticBoundaryIntegratorFactory.hh"
#include "KOpenCLBoundaryIntegralMatrix.hh"
#include "KOpenCLBoundaryIntegralVector.hh"
#include "KOpenCLBoundaryIntegralSolutionVector.hh"
#include "KRobinHood_OpenCL.hh"
#ifdef KEMFIELD_USE_MPI
#include "KRobinHood_MPI_OpenCL.hh"
#endif
#endif

using namespace KEMField;

int main(int /*argc*/, char** /*argv*/)
{
    /*
    // define the rectangle that will be defined 4 ways

    // KOpenCLSurfaceData oclSurfaceData(surfaceContainer);
    // oclSurfaceData.SetNLocal(128);
    // oclSurfaceData.ConstructOpenCLBuffers();

    double a = 1.5;
    double b = 1.3;
    KThreeVector p0(0.,0.,0.);
    KThreeVector n1(1./sqrt(2.),1./sqrt(2.),0.);
    KThreeVector n2(1./sqrt(2.),-1./sqrt(2.),0.);

    double dirichletValue = 10.2;

    double chargeDensity = 4.8;

    KSurface<KElectrostaticBasis,
    KDirichletBoundary,
    KRectangle>* t = new KSurface<KElectrostaticBasis,
    KDirichletBoundary,
    KRectangle>();

    t->SetA(a);
    t->SetB(b);
    t->SetP0(p0);
    t->SetN1(n1);
    t->SetN2(n2);

    t->SetBoundaryValue(dirichletValue);

    t->SetSolution(chargeDensity);

    KOpenCLElectrostaticBoundaryIntegrator integrator;

    double value = 0.;
    integrator.Evaluate(t->GetShape(),t,value);

    std::cout<<"value: "<<value<<std::endl;

    return 0;
    */

    /*
      KThreeVector p0(0.,0.,0.);
      KThreeVector p1(1.,0.,1.);

      double dirichletValue = 10.2;

      double chargeDensity = 4.8;

      KSurface<KElectrostaticBasis,
      KDirichletBoundary,
      KConicSection>* c = new KSurface<KElectrostaticBasis,
      KDirichletBoundary,
      KConicSection>();

      c->SetP0(p0);
      c->SetP1(p1);

      c->SetBoundaryValue(dirichletValue);

      c->SetSolution(chargeDensity);

      KOpenCLElectrostaticBoundaryIntegrator integrator;

      KElectrostaticBoundaryIntegrator integrator2;

      double value = integrator.BoundaryIntegral(c,c,0);

      std::cout<<"value: "<<value<<std::endl;

      value = integrator2.BoundaryIntegral(c,c,0);

      std::cout<<"value: "<<value<<std::endl;
    */

    /*
    double a_ = 1.5;
    double b_ = 1.3;
    KThreeVector p0(0.,0.,0.);
    KThreeVector n1(1./sqrt(2.),1./sqrt(2.),0.);
    KThreeVector n2(1./sqrt(2.),-1./sqrt(2.),0.);

    double dirichletValue = 10.2;

    KSurface<KElectrostaticBasis,
           KDirichletBoundary,
           KTriangle>* t = new KSurface<KElectrostaticBasis,
                       KDirichletBoundary,
                       KTriangle>();

    t->SetA(a_);
    t->SetB(b_);
    t->SetP0(p0);
    t->SetN1(n1);
    t->SetN2(n2);

    t->SetBoundaryValue(dirichletValue);

    KSurface<KElectrostaticBasis,
           KNeumannBoundary,
           KTriangle>* t2 = new KSurface<KElectrostaticBasis,
                        KNeumannBoundary,
                        KTriangle>();

    t2->SetA(a_);
    t2->SetB(b_);
    t2->SetP0(KPosition(2.,2.,0.));
    t2->SetN1(n1.Cross(n2).Unit());
    t2->SetN2(n2);

    t2->SetNormalBoundaryFlux(dirichletValue);

    KSurfaceContainer surfaceContainer;

    surfaceContainer.push_back(t);
    surfaceContainer.push_back(t2);

    KSurfaceContainer surfaceContainer2;
    KSurface<KElectrostaticBasis,
           KDirichletBoundary,
           KTriangle>* t3 = new KSurface<KElectrostaticBasis,
                        KDirichletBoundary,
                        KTriangle>(*t);
    KSurface<KElectrostaticBasis,
           KNeumannBoundary,
           KTriangle>* t4 = new KSurface<KElectrostaticBasis,
                        KNeumannBoundary,
                        KTriangle>(*t2);

    surfaceContainer2.push_back(t3);
    surfaceContainer2.push_back(t4);

  #ifdef KEMFIELD_USE_OPENCL
    KOpenCLSurfaceContainer oclSurfaceContainer(surfaceContainer);
    KOpenCLElectrostaticBoundaryIntegrator integrator(oclSurfaceContainer);
    KBoundaryIntegralMatrix<KOpenCLBoundaryIntegrator<KElectrostaticBasis> > A(oclSurfaceContainer,integrator);
    KBoundaryIntegralVector<KOpenCLBoundaryIntegrator<KElectrostaticBasis> > b(oclSurfaceContainer,integrator);
    KBoundaryIntegralSolutionVector<KOpenCLBoundaryIntegrator<KElectrostaticBasis> > x(oclSurfaceContainer,integrator);
  #else
    KElectrostaticBoundaryIntegrator integrator;
    KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A(surfaceContainer,integrator);
    KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x(surfaceContainer,integrator);
    KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b(surfaceContainer,integrator);
  #endif

    double accuracy = 1.e-8;

  #if defined(KEMFIELD_USE_MPI) && defined(KEMFIELD_USE_OPENCL)
    KRobinHood<KElectrostaticBoundaryIntegrator::ValueType,
           KRobinHood_MPI_OpenCL> robinHood;
  #ifndef KEMFIELD_USE_DOUBLE_PRECISION
    robinHood.SetTolerance((accuracy > 1.e-5 ? accuracy : 1.e-5));
  #else
    robinHood.SetTolerance(accuracy);
  #endif
  #elif defined(KEMFIELD_USE_MPI)
    KRobinHood<KElectrostaticBoundaryIntegrator::ValueType,
           KRobinHood_MPI> robinHood;
  #elif defined(KEMFIELD_USE_OPENCL)
    KRobinHood<KElectrostaticBoundaryIntegrator::ValueType,
           KRobinHood_OpenCL> robinHood;
  #ifndef KEMFIELD_USE_DOUBLE_PRECISION
    robinHood.SetTolerance((accuracy > 1.e-5 ? accuracy : 1.e-5));
  #else
    robinHood.SetTolerance(accuracy);
  #endif
  #else
    KRobinHood<KElectrostaticBoundaryIntegrator::ValueType> robinHood;
  #endif

    robinHood.SetResidualCheckInterval(1);
    robinHood.Solve(A,x,b);

    for (unsigned int i=0;i<A.Dimension();i++)
      for (unsigned int j=0;j<A.Dimension();j++)
        std::cout<<"A("<<i<<","<<j<<"): "<<A(i,j)<<std::endl;

    KElectrostaticBoundaryIntegrator integrator2;
    KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A2(surfaceContainer2,integrator2);
    KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x2(surfaceContainer2,integrator2);
    KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b2(surfaceContainer2,integrator2);

    for (unsigned int i=0;i<A.Dimension();i++)
      for (unsigned int j=0;j<A.Dimension();j++)
        std::cout<<"A2("<<i<<","<<j<<"): "<<A2(i,j)<<std::endl;
    */

        KSurface <KElectrostaticBasis,
        KDirichletBoundary,
        KLineSegment> *w = new KSurface<KElectrostaticBasis,
                                        KDirichletBoundary,
                                        KLineSegment>();

        w->SetP0(KThreeVector(-0.457222, 0.0504778, -0.51175));
        w->SetP1(KThreeVector(-0.463342, 0.0511534, -0.515712));
        w->SetDiameter(0.0003);
        w->SetBoundaryValue(-900);

        KSurfaceContainer surfaceContainer;
        surfaceContainer.push_back(w);
        
	KOpenCLSurfaceContainer oclSurfaceContainer(surfaceContainer);
        KOpenCLElectrostaticBoundaryIntegrator integrator{KoclEBIFactory::MakeDefault(oclSurfaceContainer)};
	KBoundaryIntegralMatrix <KOpenCLBoundaryIntegrator<KElectrostaticBasis>> A(oclSurfaceContainer, integrator);
        KBoundaryIntegralVector <KOpenCLBoundaryIntegrator<KElectrostaticBasis>> b(oclSurfaceContainer, integrator);
        KBoundaryIntegralSolutionVector <KOpenCLBoundaryIntegrator<KElectrostaticBasis>> x(oclSurfaceContainer, integrator);
        
	KElectrostaticBoundaryIntegrator integrator2{KEBIFactory::MakeDefault()};
        KBoundaryIntegralMatrix <KElectrostaticBoundaryIntegrator> A2(surfaceContainer, integrator2);
        KBoundaryIntegralSolutionVector <KElectrostaticBoundaryIntegrator> x2(surfaceContainer, integrator2);
        KBoundaryIntegralVector <KElectrostaticBoundaryIntegrator> b2(surfaceContainer, integrator2);
	oclSurfaceContainer.ConstructOpenCLObjects();

        std::cout << "Test: " << A(0, 0) << std::endl;

    return 0;
}
