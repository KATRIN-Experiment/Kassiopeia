/*
 * CubeCapacitanceTest.cc
 *
 *  Created on: 22 Oct 2020
 *      Author: jbehrens
 *
 *  Based on TestCubeCapacitance.cc
 */

#include "KElectrostaticBoundaryIntegratorFactory.hh"
#include "KBoundaryIntegralMatrix.hh"
#include "KBoundaryIntegralSolutionVector.hh"
#include "KBoundaryIntegralVector.hh"
#include "KBoundaryMatrixGenerator.hh"
#include "KGBEM.hh"
#include "KGBEMConverter.hh"
#include "KGMesher.hh"
#include "KGRotatedObject.hh"
#include "KGaussianElimination.hh"
#include "KRobinHood.hh"
#include "KBiconjugateGradientStabilized.hh"
#include "KGeneralizedMinimalResidual.hh"
#include "KSimpleIterativeKrylovSolver.hh"
#include "KSurface.hh"
#include "KSurfaceContainer.hh"
#include "KSurfaceTypes.hh"
#include "KTypelist.hh"

#include "KEMConstants.hh"

#ifdef KEMFIELD_USE_OPENCL
#include "KOpenCLElectrostaticBoundaryIntegratorFactory.hh"
#include "KOpenCLsurfaceContainer->hh"
#include "KOpenCLBoundaryIntegralMatrix.hh"
#include "KOpenCLBoundaryIntegralSolutionVector.hh"
#include "KOpenCLBoundaryIntegralVector.hh"
#include "KOpenCLElectrostaticBoundaryIntegratorFactory.hh"
#include "KOpenCLsurfaceContainer->hh"
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

class KEMFieldCubeTest : public KEMFieldTest
{

protected:
    void SetUp() override
    {
        KEMFieldTest::SetUp();

        using namespace KGeoBag;

        surfaceContainer = new KSurfaceContainer();

        int scale = 8;
        double power = 1.5;

        // Construct the shape
        KGBox* box = new KGBox();

        box->SetX0(-.5);
        box->SetX1(.5);
        box->SetXMeshCount(scale);
        box->SetXMeshPower(power);

        box->SetY0(-.5);
        box->SetY1(.5);
        box->SetYMeshCount(scale);
        box->SetYMeshPower(power);

        box->SetZ0(-.5);
        box->SetZ1(.5);
        box->SetZMeshCount(scale);
        box->SetZMeshPower(power);

        KGSurface* cube = new KGSurface(box);
        cube->SetName("box");
        cube->MakeExtension<KGMesh>();
        cube->MakeExtension<KGElectrostaticDirichlet>();
        cube->AsExtension<KGElectrostaticDirichlet>()->SetBoundaryValue(1.);

        // Mesh the elements
        mesher = new KGMesher();
        cube->AcceptNode(mesher);

        geometryConverter = new KGBEMMeshConverter(*surfaceContainer);
        cube->AcceptNode(geometryConverter);

        MPI_SINGLE_PROCESS
        std::cout << "Discretized cube has " << surfaceContainer->size() << " elements" << std::endl;
    }

    void TearDown() override
    {
        double tol = 1e-2;  // depends on discretization scale
        double Q = 0.;

        for (KSurfaceContainer::iterator it = surfaceContainer->begin(); it != surfaceContainer->end(); it++) {
            Q += (dynamic_cast<KRectangle*>(*it)->Area() * dynamic_cast<KElectrostaticBasis*>(*it)->GetSolution());
        }

        double C = Q / (4. * M_PI * KEMConstants::Eps0);

        double C_Read = 0.6606785;
        double C_Read_err = 0.0000006;

        MPI_SINGLE_PROCESS {
            std::cout << std::setprecision(7) << "Capacitance:    " << C << std::endl;
            std::cout.setf(std::ios::fixed, std::ios::floatfield);
            std::cout << std::setprecision(7) << "Accepted value: " << C_Read << " +\\- " << C_Read_err << std::endl;
            std::cout << "Accuracy:       " << (fabs(C - C_Read) / C_Read) * 100 << " %" << std::endl;
        }

        ASSERT_NEAR(C_Read, C, tol);

        KEMFieldTest::TearDown();
    }

    KSurfaceContainer* surfaceContainer;
    KGeoBag::KGMesher* mesher;
    KGeoBag::KGBEMMeshConverter* geometryConverter;
};

TEST_F(KEMFieldCubeTest, CubeCapacitance_GaussAnalytic)
{
    // method 0 = gauss; integrator type 0 = analytic
    KElectrostaticBoundaryIntegrator integrator = KEBIFactory::MakeAnalytic();
    KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A(*surfaceContainer, integrator);
    KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x(*surfaceContainer, integrator);
    KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b(*surfaceContainer, integrator);

    KGaussianElimination<KElectrostaticBoundaryIntegrator::ValueType> gaussianElimination;

    gaussianElimination.Solve(A, x, b);
}

TEST_F(KEMFieldCubeTest, CubeCapacitance_GaussRWG)
{
    // method 0 = gauss; integrator type 1 = RWG
    KElectrostaticBoundaryIntegrator integrator = KEBIFactory::MakeRWG();
    KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A(*surfaceContainer, integrator);
    KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x(*surfaceContainer, integrator);
    KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b(*surfaceContainer, integrator);

    KGaussianElimination<KElectrostaticBoundaryIntegrator::ValueType> gaussianElimination;

    gaussianElimination.Solve(A, x, b);
}

TEST_F(KEMFieldCubeTest, CubeCapacitance_GaussNumeric)
{
    // method 0 = gauss; integrator type 2 = numeric
    KElectrostaticBoundaryIntegrator integrator = KEBIFactory::MakeNumeric();
    KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A(*surfaceContainer, integrator);
    KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x(*surfaceContainer, integrator);
    KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b(*surfaceContainer, integrator);

    KGaussianElimination<KElectrostaticBoundaryIntegrator::ValueType> gaussianElimination;

    gaussianElimination.Solve(A, x, b);
}

TEST_F(KEMFieldCubeTest, DiskCapacitance_RobinHoodRWG)
{
    // method 1 = robin hood; integrator type 1 = RWG
    double accuracy = 1.e-4;
    int increment = 100;

    KElectrostaticBoundaryIntegrator integrator = KEBIFactory::MakeRWG();
    KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A(*surfaceContainer, integrator);
    KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x(*surfaceContainer, integrator);
    KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b(*surfaceContainer, integrator);

    KRobinHood<KElectrostaticBoundaryIntegrator::ValueType> robinHood;

    robinHood.SetTolerance(accuracy);
    robinHood.SetResidualCheckInterval(increment);
    robinHood.Solve(A, x, b);
}

TEST_F(KEMFieldCubeTest, CubeCapacitance_RobinHoodNumeric)
{
    // method 1 = robin hood; integrator type 2 = numeric
    double accuracy = 1.e-4;
    int increment = 100;

    KElectrostaticBoundaryIntegrator integrator = KEBIFactory::MakeNumeric();
    KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A(*surfaceContainer, integrator);
    KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x(*surfaceContainer, integrator);
    KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b(*surfaceContainer, integrator);

    KRobinHood<KElectrostaticBoundaryIntegrator::ValueType> robinHood;

    robinHood.SetTolerance(accuracy);
    robinHood.SetResidualCheckInterval(increment);
    robinHood.Solve(A, x, b);
}

#ifdef KEMFIELD_USE_OPENCL
TEST_F(KEMFieldCubeTest, CubeCapacitance_RobinHoodRWG_OpenCL)
{
    // method 1 = robin hood; integrator type 1 = RWG
    double accuracy = 1.e-4;
    int increment = 100;

    KOpenCLSurfaceContainer* oclSurfaceContainer = new KOpenCLSurfaceContainer(surfaceContainer);
    KOpenCLElectrostaticBoundaryIntegrator integrator = KoclEBIFactory::MakeRWG(oclSurfaceContainer);
    KBoundaryIntegralMatrix<KOpenCLBoundaryIntegrator<KElectrostaticBasis>> A(*oclSurfaceContainer, integrator);
    KBoundaryIntegralVector<KOpenCLBoundaryIntegrator<KElectrostaticBasis>> b(*oclSurfaceContainer, integrator);
    KBoundaryIntegralSolutionVector<KOpenCLBoundaryIntegrator<KElectrostaticBasis>> x(*oclSurfaceContainer, integrator);

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

/** FIXME - library dependencies are broken in build **/
#if 0
TEST_F(KEMFieldCubeTest, CubeCapacitance_KrylovNumeric_BiCGSTAB)
{
    // method = krylov; integrator type 2 = numeric
    double accuracy = 1.e-8;

    KElectrostaticBoundaryIntegrator integrator = KEBIFactory::MakeNumeric();
    KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A(*surfaceContainer, integrator);
    KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x*(surfaceContainer, integrator);
    KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b(*surfaceContainer, integrator);

    KSimpleIterativeKrylovSolver<KElectrostaticBoundaryIntegrator::ValueType, KBiconjugateGradientStabilized> biCGStabSolver;

    biCGStabSolver.SetTolerance(accuracy);
    biCGStabSolver.Solve(A, x, b);
}

TEST_F(KEMFieldCubeTest, CubeCapacitance_KrylovNumeric_GMRes)
{
    // method = krylov; integrator type 2 = numeric
    double accuracy = 1.e-8;

    KElectrostaticBoundaryIntegrator integrator = KEBIFactory::MakeNumeric();
    KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A(*surfaceContainer, integrator);
    KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x(*surfaceContainer, integrator);
    KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b(*surfaceContainer, integrator);

    KSimpleIterativeKrylovSolver<KElectrostaticBoundaryIntegrator::ValueType, KGeneralizedMinimalResidual> gmresSolver;

    gmresSolver.SetTolerance(accuracy);
    gmresSolver.Solve(A, x, b);
}
#endif

#ifdef KEMFIELD_USE_PETSC
TEST_F(KEMFieldCubeTest, CubeCapacitance_PETSc)
{
    // method 2 = PETSc
    KElectrostaticBoundaryIntegrator integrator{KEBIFactory::MakeNumeric()};
    KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A(*surfaceContainer, integrator);
    KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x(*surfaceContainer, integrator);
    KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b(*surfaceContainer, integrator);

    KPETScSolver<KElectrostaticBoundaryIntegrator::ValueType> petscSolver;

    petscSolver.SetTolerance(accuracy);
    petscSolver.Solve(A, x, b);
}
#endif
