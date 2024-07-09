/*
 * CapacitanceTest.cc
 *
 *  Created on: 22 Oct 2020
 *      Author: jbehrens
 *
 *  Based on TestCapacitance.cc
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

class KEMFieldDiskTest : public KEMFieldTest
{

protected:
    void SetUp() override
    {
        KEMFieldTest::SetUp();

        using namespace KGeoBag;

        surfaceContainer = new KSurfaceContainer();

        int scale = 128;
        double power = 1.5;

        std::vector<double> segments(2 * scale);
        DiscretizeInterval(2 * 1., 2 * scale, power, segments);

        typedef KSurface<KElectrostaticBasis, KDirichletBoundary, KConicSection> KEMConicSection;

        double r0 = 1.;
        double r1 = 1.;
        for (int i = 0; i < scale; i++) {
            r1 -= segments.at(i);

            KEMConicSection* cs = new KEMConicSection();
            cs->SetR0(r0);
            cs->SetZ0(0.);
            cs->SetR1(r1);
            cs->SetZ1(0.);
            cs->SetBoundaryValue(1.);

            surfaceContainer->push_back(cs);

            r0 = r1;
        }

        MPI_SINGLE_PROCESS
        std::cout << "Discretized disk has " << surfaceContainer->size() << " elements" << std::endl;
    }

    void TearDown() override
    {
        typedef KSurface<KElectrostaticBasis, KDirichletBoundary, KConicSection> KEMConicSection;

        double tol = 1e-2;  // depends on discretization scale
        double Q = 0.;

        for (KSurfaceContainer::iterator it = surfaceContainer->begin(); it != surfaceContainer->end(); it++) {
            Q += (static_cast<KEMConicSection*>(*it)->Area() * static_cast<KEMConicSection*>(*it)->GetSolution());
        }

        double C = Q / KEMConstants::Eps0;
        double C_analytic = 8.;

        MPI_SINGLE_PROCESS {
            std::cout << std::setprecision(7) << "Capacitance:    " << C << std::endl;
            std::cout.setf(std::ios::fixed, std::ios::floatfield);
            std::cout << std::setprecision(7) << "Analytic value: " << C_analytic << std::endl;
            std::cout << "Accuracy:       " << (fabs(C - C_analytic) / C_analytic) * 100 << " %" << std::endl;
        }

        ASSERT_NEAR(C_analytic, C, tol);

        KEMFieldTest::TearDown();
    }

    KSurfaceContainer* surfaceContainer;

private:
    static void DiscretizeInterval(double interval, int nSegments, double power, std::vector<double>& segments)
    {
        if (nSegments == 1)
            segments[0] = interval;
        else {
            double inc1, inc2;
            double mid = interval * .5;
            if (nSegments % 2 == 1) {
                segments[nSegments / 2] = interval / nSegments;
                mid -= interval / (2 * nSegments);
            }

            for (int i = 0; i < nSegments / 2; i++) {
                inc1 = ((double) i) / (nSegments / 2.);
                inc2 = ((double) (i + 1)) / (nSegments / 2.);

                inc1 = pow(inc1, power);
                inc2 = pow(inc2, power);

                segments[i] = segments[nSegments - (i + 1)] = mid * (inc2 - inc1);
            }
        }
        return;
    }

};

TEST_F(KEMFieldDiskTest, Capacitance_GaussAnalytic)
{
    // method 0 = gauss; integrator type 0 = analytic
    KElectrostaticBoundaryIntegrator integrator = KEBIFactory::MakeAnalytic();
    KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A(*surfaceContainer, integrator);
    KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x(*surfaceContainer, integrator);
    KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b(*surfaceContainer, integrator);

    KGaussianElimination<KElectrostaticBoundaryIntegrator::ValueType> gaussianElimination;

    gaussianElimination.Solve(A, x, b);
}

TEST_F(KEMFieldDiskTest, Capacitance_GaussRWG)
{
    // method 0 = gauss; integrator type 1 = RWG
    KElectrostaticBoundaryIntegrator integrator = KEBIFactory::MakeRWG();
    KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A(*surfaceContainer, integrator);
    KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x(*surfaceContainer, integrator);
    KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b(*surfaceContainer, integrator);

    KGaussianElimination<KElectrostaticBoundaryIntegrator::ValueType> gaussianElimination;

    gaussianElimination.Solve(A, x, b);
}

TEST_F(KEMFieldDiskTest, Capacitance_GaussNumeric)
{
    // method 0 = gauss; integrator type 2 = numeric
    KElectrostaticBoundaryIntegrator integrator = KEBIFactory::MakeNumeric();
    KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A(*surfaceContainer, integrator);
    KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x(*surfaceContainer, integrator);
    KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b(*surfaceContainer, integrator);

    KGaussianElimination<KElectrostaticBoundaryIntegrator::ValueType> gaussianElimination;

    gaussianElimination.Solve(A, x, b);
}

TEST_F(KEMFieldDiskTest, Capacitance_RobinHoodRWG)
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

TEST_F(KEMFieldDiskTest, Capacitance_RobinHoodNumeric)
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
TEST_F(KEMFieldDiskTest, Capacitance_RobinHoodRWG_OpenCL)
{
    // method 1 = robin hood; integrator type 1 = RWG
    double accuracy = 1.e-4;
    int increment = 100;

    KOpenCLSurfaceContainer* oclSurfaceContainer = new KOpenCLSurfaceContainer(*surfaceContainer);
    KOpenCLElectrostaticBoundaryIntegrator integrator = KoclEBIFactory::MakeRWG(*oclSurfaceContainer);
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

#ifdef KEMFIELD_USE_PETSC
TEST_F(KEMFieldDiskTest, Capacitance_PETSc)
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
