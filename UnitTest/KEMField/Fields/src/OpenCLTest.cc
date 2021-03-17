/*
 * OpenCLTest.cc
 *
 *  Created on: 11 Nov 2020
 *      Author: jbehrens
 *
 *  Based on TestOpenCLPlugin.cc
 */

#include "KEMConstants.hh"
#include "KEMFieldTest.hh"

#include "KBoundaryIntegralMatrix.hh"
#include "KElectrostaticBoundaryIntegratorFactory.hh"
#include "KSurface.hh"
#include "KSurfaceContainer.hh"
#include "KSurfaceTypes.hh"
#include "KOpenCLBoundaryIntegralMatrix.hh"
#include "KOpenCLBoundaryIntegralSolutionVector.hh"
#include "KOpenCLBoundaryIntegralVector.hh"
#include "KOpenCLElectrostaticBoundaryIntegratorFactory.hh"
#include "KOpenCLSurfaceContainer.hh"
#include "KRobinHood.hh"
#ifdef KEMFIELD_USE_MPI
#include "KRobinHood_MPI_OpenCL.hh"
#else
#include "KRobinHood_OpenCL.hh"
#endif

using namespace KEMField;

class KEMFieldOpenCLTest : public KEMFieldTest
{};

TEST_F(KEMFieldOpenCLTest, Triangle)
{
    double a_ = 1.5;
    double b_ = 1.3;
    KFieldVector p0(0.,0.,0.);
    KFieldVector n1(1./sqrt(2.),1./sqrt(2.),0.);
    KFieldVector n2(1./sqrt(2.),-1./sqrt(2.),0.);

    double dirichletValue = 10.2;

    auto* t1 = new KSurface<KElectrostaticBasis, KDirichletBoundary, KTriangle>();

    t1->SetA(a_);
    t1->SetB(b_);
    t1->SetP0(p0);
    t1->SetN1(n1);
    t1->SetN2(n2);

    t1->SetBoundaryValue(dirichletValue);

    auto* t2 = new KSurface<KElectrostaticBasis, KNeumannBoundary, KTriangle>();

    t2->SetA(a_);
    t2->SetB(b_);
    t2->SetP0(KPosition(2.,2.,0.));
    t2->SetN1(n1.Cross(n2).Unit());
    t2->SetN2(n2);

    t2->SetNormalBoundaryFlux(dirichletValue);

    KSurfaceContainer surfaceContainer;

    surfaceContainer.push_back(t1);
    surfaceContainer.push_back(t2);

    KSurfaceContainer surfaceContainer2;

    auto* t3 = new KSurface<KElectrostaticBasis, KDirichletBoundary, KTriangle>(*t1);
    auto* t4 = new KSurface<KElectrostaticBasis, KNeumannBoundary, KTriangle>(*t2);

    surfaceContainer2.push_back(t3);
    surfaceContainer2.push_back(t4);

    KOpenCLSurfaceContainer oclSurfaceContainer(surfaceContainer);
    KOpenCLElectrostaticBoundaryIntegrator integrator{KoclEBIFactory::MakeDefault(oclSurfaceContainer)};
    KBoundaryIntegralMatrix<KOpenCLBoundaryIntegrator<KElectrostaticBasis> > A1(oclSurfaceContainer,integrator);
    KBoundaryIntegralVector<KOpenCLBoundaryIntegrator<KElectrostaticBasis> > b1(oclSurfaceContainer,integrator);
    KBoundaryIntegralSolutionVector<KOpenCLBoundaryIntegrator<KElectrostaticBasis> > x1(oclSurfaceContainer,integrator);

#ifndef KEMFIELD_USE_DOUBLE_PRECISION
    double accuracy = 1.e-8;
#else
    double accuracy = 1.e-5;
#endif  // KEMFIELD_USE_DOUBLE_PRECISION

#ifdef KEMFIELD_USE_MPI
    KRobinHood<KElectrostaticBoundaryIntegrator::ValueType, KRobinHood_MPI_OpenCL> robinHood;
#else
    KRobinHood<KElectrostaticBoundaryIntegrator::ValueType, KRobinHood_OpenCL> robinHood;
#endif  // KEMFIELD_USE_MPI

    oclSurfaceContainer.ConstructOpenCLObjects();

    robinHood.SetTolerance(accuracy);
    robinHood.SetResidualCheckInterval(1);
    robinHood.Solve(A1, x1, b1);

    KRobinHood<KElectrostaticBoundaryIntegrator::ValueType> robinHood2;

    KElectrostaticBoundaryIntegrator integrator2{KEBIFactory::MakeDefault()};
    KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A2(surfaceContainer2,integrator2);
    KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x2(surfaceContainer2,integrator2);
    KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b2(surfaceContainer2,integrator2);

    robinHood2.SetTolerance(accuracy);
    robinHood2.SetResidualCheckInterval(1);
    robinHood2.Solve(A2, x2, b2);

    for (unsigned int i=0;i<A1.Dimension();i++)
      for (unsigned int j=0;j<A1.Dimension();j++)
        ASSERT_NEAR(A1(i,j), A2(i,j), accuracy*fabs(A2(i,j)));

    for (unsigned int i=0;i<b1.Dimension();i++)
        ASSERT_NEAR(b1(i), b2(i), accuracy*fabs(b2(i)));

    for (unsigned int i=0;i<x1.Dimension();i++)
        ASSERT_NEAR(x1(i), x2(i), accuracy*fabs(x2(i)));
}

TEST_F(KEMFieldOpenCLTest, LineSegment)
{
    auto* w1 = new KSurface<KElectrostaticBasis, KDirichletBoundary, KLineSegment>();

    w1->SetP0(KFieldVector(-0.457222, 0.0504778, -0.51175));
    w1->SetP1(KFieldVector(-0.463342, 0.0511534, -0.515712));
    w1->SetDiameter(0.0003);

    w1->SetBoundaryValue(-900);

    KSurfaceContainer surfaceContainer;
    surfaceContainer.push_back(w1);

    auto* w2 = new KSurface<KElectrostaticBasis, KDirichletBoundary, KLineSegment>(*w1);

    KSurfaceContainer surfaceContainer2;
    surfaceContainer2.push_back(w2);

    KOpenCLSurfaceContainer oclSurfaceContainer(surfaceContainer);
    KOpenCLElectrostaticBoundaryIntegrator integrator{KoclEBIFactory::MakeDefault(oclSurfaceContainer)};
    KBoundaryIntegralMatrix<KOpenCLBoundaryIntegrator<KElectrostaticBasis> > A1(oclSurfaceContainer,integrator);
    KBoundaryIntegralVector<KOpenCLBoundaryIntegrator<KElectrostaticBasis> > b1(oclSurfaceContainer,integrator);
    KBoundaryIntegralSolutionVector<KOpenCLBoundaryIntegrator<KElectrostaticBasis> > x1(oclSurfaceContainer,integrator);

#ifndef KEMFIELD_USE_DOUBLE_PRECISION
    double accuracy = 1.e-8;
#else
    double accuracy = 1.e-5;
#endif  // KEMFIELD_USE_DOUBLE_PRECISION

#ifdef KEMFIELD_USE_MPI
    KRobinHood<KElectrostaticBoundaryIntegrator::ValueType, KRobinHood_MPI_OpenCL> robinHood;
#else
    KRobinHood<KElectrostaticBoundaryIntegrator::ValueType, KRobinHood_OpenCL> robinHood;
#endif  // KEMFIELD_USE_MPI

    oclSurfaceContainer.ConstructOpenCLObjects();

    robinHood.SetTolerance(accuracy);
    robinHood.SetResidualCheckInterval(1);
    robinHood.Solve(A1, x1, b1);

    KRobinHood<KElectrostaticBoundaryIntegrator::ValueType> robinHood2;

    KElectrostaticBoundaryIntegrator integrator2{KEBIFactory::MakeDefault()};
    KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A2(surfaceContainer2,integrator2);
    KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x2(surfaceContainer2,integrator2);
    KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b2(surfaceContainer2,integrator2);

    robinHood2.SetTolerance(accuracy);
    robinHood2.SetResidualCheckInterval(1);
    robinHood2.Solve(A2, x2, b2);

    for (unsigned int i=0;i<A1.Dimension();i++)
      for (unsigned int j=0;j<A1.Dimension();j++)
        ASSERT_NEAR(A1(i,j), A2(i,j), accuracy*fabs(A2(i,j)));

    for (unsigned int i=0;i<b1.Dimension();i++)
        ASSERT_NEAR(b1(i), b2(i), accuracy*fabs(b2(i)));

    for (unsigned int i=0;i<x1.Dimension();i++)
        ASSERT_NEAR(x1(i), x2(i), accuracy*fabs(x2(i)));
}
