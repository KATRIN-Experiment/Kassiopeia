/*
 * SphereCapacitanceTest.cc
 *
 *  Created on: 22 Oct 2020
 *      Author: jbehrens
 *
 *  Based on TestSphereCapacitance.cc
 */

#include "KElectrostaticBoundaryIntegratorFactory.hh"
#include "KBoundaryIntegralMatrix.hh"
#include "KBoundaryIntegralSolutionVector.hh"
#include "KBoundaryIntegralVector.hh"
#include "KGBEM.hh"
#include "KGBEMConverter.hh"
#include "KGMesher.hh"
#include "KGRotatedObject.hh"
#include "KGaussianElimination.hh"
#include "KRobinHood.hh"
#include "KSurface.hh"
#include "KSurfaceContainer.hh"
#include "KSurfaceTypes.hh"
#include "KTypelist.hh"

#include "KEMConstants.hh"

#ifdef KEMFIELD_USE_OPENCL
#include "KOpenCLElectrostaticBoundaryIntegratorFactory.hh"
#include "KOpenCLSurfaceContainer.hh"
#include "KOpenCLBoundaryIntegralMatrix.hh"
#include "KOpenCLBoundaryIntegralSolutionVector.hh"
#include "KOpenCLBoundaryIntegralVector.hh"
#include "KOpenCLElectrostaticBoundaryIntegratorFactory.hh"
#include "KOpenCLSurfaceContainer.hh"
#ifdef KEMFIELD_USE_MPI
#include "KRobinHood_MPI_OpenCL.hh"
#else
#include "KRobinHood_OpenCL.hh"
#endif
#endif

#ifdef KEMFIELD_USE_PETSC
#include "KPETScInterface.hh"
#include "KPETScSolver.hh"
#endif

#include "KEMFieldTest.hh"

using namespace KEMField;

class KEMFieldSphereTest : public KEMFieldTest
{
protected:
    void SetUp() override
    {
        KEMFieldTest::SetUp();

        using namespace KGeoBag;

        int scale = 24;

        // Construct the shape
        double p1[2], p2[2];
        double radius = 1.;
        KGRotatedObject* hemi1 = new KGRotatedObject(scale, scale/2);
        p1[0] = -1.;
        p1[1] = 0.;
        p2[0] = 0.;
        p2[1] = 1.;
        hemi1->AddArc(p2, p1, radius, true);

        KGRotatedObject* hemi2 = new KGRotatedObject(scale, scale/2);
        p2[0] = 1.;
        p2[1] = 0.;
        p1[0] = 0.;
        p1[1] = 1.;
        hemi2->AddArc(p1, p2, radius, false);

        // Construct shape placement
        KGRotatedSurface* h1 = new KGRotatedSurface(hemi1);
        KGSurface* hemisphere1 = new KGSurface(h1);
        hemisphere1->SetName("hemisphere1");
        hemisphere1->MakeExtension<KGMesh>();
        hemisphere1->MakeExtension<KGElectrostaticDirichlet>();
        hemisphere1->AsExtension<KGElectrostaticDirichlet>()->SetBoundaryValue(1.);

        KGRotatedSurface* h2 = new KGRotatedSurface(hemi2);
        KGSurface* hemisphere2 = new KGSurface(h2);
        hemisphere2->SetName("hemisphere2");
        hemisphere2->MakeExtension<KGMesh>();
        hemisphere2->MakeExtension<KGElectrostaticDirichlet>();
        hemisphere2->AsExtension<KGElectrostaticDirichlet>()->SetBoundaryValue(1.);

        // Mesh the elements
        KGMesher* mesher = new KGMesher();
        hemisphere1->AcceptNode(mesher);
        hemisphere2->AcceptNode(mesher);

        KGBEMMeshConverter geometryConverter(surfaceContainer);
        geometryConverter.SetMinimumArea(1.e-12);
        hemisphere1->AcceptNode(&geometryConverter);
        hemisphere2->AcceptNode(&geometryConverter);

        MPI_SINGLE_PROCESS
        std::cout << "Discretized sphere has " << surfaceContainer.size() << " elements" << std::endl;
    }

    void TearDown() override
    {
        double tol = 5e-2;  // depends on discretization scale
        double Q = 0.;

        for (KSurfaceContainer::iterator it = surfaceContainer.begin(); it != surfaceContainer.end(); it++) {
            Q += (dynamic_cast<KTriangle*>(*it)->Area() * dynamic_cast<KElectrostaticBasis*>(*it)->GetSolution());
        }

        double C = Q / (4. * M_PI * KEMConstants::Eps0);

        double C_Analytic = 1.;

        MPI_SINGLE_PROCESS {
            std::cout << std::setprecision(7) << "Capacitance:    " << C << std::endl;
            std::cout.setf(std::ios::fixed, std::ios::floatfield);
            std::cout << std::setprecision(7) << "Accepted value: " << C_Analytic << std::endl;
            std::cout << "Accuracy:       " << (fabs(C - C_Analytic) / C_Analytic) * 100 << " %" << std::endl;
        }

        ASSERT_NEAR(C_Analytic, C, tol);

        KEMFieldTest::TearDown();
    }

    KSurfaceContainer surfaceContainer;

};

TEST_F(KEMFieldSphereTest, SphereCapacitance_GaussAnalytic)
{
    // method 0 = gauss; integrator type 0 = analytic
    KElectrostaticBoundaryIntegrator integrator = KEBIFactory::MakeAnalytic();
    KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A(surfaceContainer, integrator);
    KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x(surfaceContainer, integrator);
    KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b(surfaceContainer, integrator);

    KGaussianElimination<KElectrostaticBoundaryIntegrator::ValueType> gaussianElimination;

    gaussianElimination.Solve(A, x, b);
}

TEST_F(KEMFieldSphereTest, SphereCapacitance_GaussRWG)
{
    // method 0 = gauss; integrator type 1 = RWG
    KElectrostaticBoundaryIntegrator integrator = KEBIFactory::MakeRWG();
    KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A(surfaceContainer, integrator);
    KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x(surfaceContainer, integrator);
    KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b(surfaceContainer, integrator);

    KGaussianElimination<KElectrostaticBoundaryIntegrator::ValueType> gaussianElimination;

    gaussianElimination.Solve(A, x, b);
}

TEST_F(KEMFieldSphereTest, SphereCapacitance_GaussNumeric)
{
    // method 0 = gauss; integrator type 2 = numeric
    KElectrostaticBoundaryIntegrator integrator = KEBIFactory::MakeNumeric();
    KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A(surfaceContainer, integrator);
    KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x(surfaceContainer, integrator);
    KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b(surfaceContainer, integrator);

    KGaussianElimination<KElectrostaticBoundaryIntegrator::ValueType> gaussianElimination;

    gaussianElimination.Solve(A, x, b);
}

TEST_F(KEMFieldSphereTest, DiskCapacitance_RobinHoodRWG)
{
    // method 1 = robin hood; integrator type 1 = RWG
    double accuracy = 1.e-4;
    int increment = 100;

    KElectrostaticBoundaryIntegrator integrator = KEBIFactory::MakeRWG();
    KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A(surfaceContainer, integrator);
    KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x(surfaceContainer, integrator);
    KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b(surfaceContainer, integrator);

    KRobinHood<KElectrostaticBoundaryIntegrator::ValueType> robinHood;

    robinHood.SetTolerance(accuracy);
    robinHood.SetResidualCheckInterval(increment);
    robinHood.Solve(A, x, b);
}

TEST_F(KEMFieldSphereTest, SphereCapacitance_RobinHoodNumeric)
{
    // method 1 = robin hood; integrator type 2 = numeric
    double accuracy = 1.e-4;
    int increment = 100;

    KElectrostaticBoundaryIntegrator integrator = KEBIFactory::MakeNumeric();
    KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A(surfaceContainer, integrator);
    KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x(surfaceContainer, integrator);
    KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b(surfaceContainer, integrator);

    KRobinHood<KElectrostaticBoundaryIntegrator::ValueType> robinHood;

    robinHood.SetTolerance(accuracy);
    robinHood.SetResidualCheckInterval(increment);
    robinHood.Solve(A, x, b);
}

#ifdef KEMFIELD_USE_OPENCL
/* FIXME: this test is broken - investigate! */
TEST_F(KEMFieldSphereTest, SphereCapacitance_RobinHoodRWG_OpenCL)
{
    // method 1 = robin hood; integrator type 1 = RWG
    double accuracy = 1.e-4;
    int increment = 100;

    KOpenCLSurfaceContainer oclSurfaceContainer(surfaceContainer);
    KOpenCLElectrostaticBoundaryIntegrator integrator = KoclEBIFactory::MakeRWG(oclSurfaceContainer);
    KBoundaryIntegralMatrix<KOpenCLBoundaryIntegrator<KElectrostaticBasis>> A(oclSurfaceContainer, integrator);
    KBoundaryIntegralVector<KOpenCLBoundaryIntegrator<KElectrostaticBasis>> b(oclSurfaceContainer, integrator);
    KBoundaryIntegralSolutionVector<KOpenCLBoundaryIntegrator<KElectrostaticBasis>> x(oclSurfaceContainer, integrator);

#ifdef KEMFIELD_USE_MPI
    KRobinHood<KElectrostaticBoundaryIntegrator::ValueType, KRobinHood_MPI_OpenCL> robinHood;
#else
    KRobinHood<KElectrostaticBoundaryIntegrator::ValueType, KRobinHood_OpenCL> robinHood;
#endif  // KEMFIELD_USE_MPI

    robinHood.SetTolerance(accuracy);
    robinHood.SetResidualCheckInterval(increment);
    robinHood.Solve(A, x, b);
}
#endif

#ifdef KEMFIELD_USE_PETSC
TEST_F(KEMFieldSphereTest, SphereCapacitance_PETSc)
{
    // method 2 = PETSc
    KElectrostaticBoundaryIntegrator integrator{KEBIFactory::MakeNumeric()};
    KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A(surfaceContainer, integrator);
    KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x(surfaceContainer, integrator);
    KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b(surfaceContainer, integrator);

    KPETScSolver<KElectrostaticBoundaryIntegrator::ValueType> petscSolver;

    petscSolver.SetTolerance(accuracy);
    petscSolver.Solve(A, x, b);
}
#endif
