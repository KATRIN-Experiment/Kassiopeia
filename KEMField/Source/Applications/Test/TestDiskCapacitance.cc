#include "KElectrostaticBoundaryIntegratorOptions.hh"
#include "KBoundaryIntegralMatrix.hh"
#include "KBoundaryIntegralSolutionVector.hh"
#include "KBoundaryIntegralVector.hh"
#include "KEMConstants.hh"
#include "KEMCout.hh"
#include "KGaussianElimination.hh"
#include "KRobinHood.hh"
#include "KSurface.hh"
#include "KSurfaceContainer.hh"
#include "KSurfaceTypes.hh"

#include <cstdlib>
#include <getopt.h>
#include <iostream>

#ifdef KEMFIELD_USE_MPI
#include "KRobinHood_MPI.hh"
#define MPI_SINGLE_PROCESS if (KMPIInterface::GetInstance()->GetProcess() == 0)
#else
#define MPI_SINGLE_PROCESS
#endif

#ifdef KEMFIELD_USE_OPENCL
#include "KOpenCLBoundaryIntegralMatrix.hh"
#include "KOpenCLBoundaryIntegralSolutionVector.hh"
#include "KOpenCLBoundaryIntegralVector.hh"
#include "KRobinHood_OpenCL.hh"
#ifdef KEMFIELD_USE_MPI
#include "KRobinHood_MPI_OpenCL.hh"
#endif
#endif

#ifdef KEMFIELD_USE_PETSC
#include "KPETScInterface.hh"
#include "KPETScSolver.hh"
#endif

#ifdef KEMFIELD_USE_VTK
#include "KVTKIterationPlotter.hh"
#endif

using namespace KEMField;

void DiscretizeInterval(double interval, int nSegments, double power, std::vector<double>& segments);

int main(int argc, char* argv[])
{
#ifdef KEMFIELD_USE_PETSC
    KPETScInterface::GetInstance()->Initialize(&argc, &argv);
#elif KEMFIELD_USE_MPI
    KMPIInterface::GetInstance()->Initialize(&argc, &argv);
#endif

    std::string usage = "\n"
                        "Usage: TestCubeCapacitance <options>\n"
                        "\n"
                        "This program computes the capacitance of the unit cube, and compares them to\n"
                        "the analytic solution of 8*epsilon0*r.\n"
                        "\n"
                        "\tAvailable options:\n"
                        "\t -h, --help               (shows this message and exits)\n"
                        "\t -v, --verbose            (0..5; sets the verbosity)\n"
                        "\t -s, --scale              (discretization scale)\n"
                        "\t -p, --power              (discretization power)\n"
                        "\t -a, --accuracy           (accuracy of charge density computation)\n"
                        "\t -i, --increment          (increment of accuracy check/print/log)\n"
#ifdef KEMFIELD_USE_VTK
                        "\t -e, --with-plot          (dynamic plot of residual norm)\n"
#endif
                        "\t -m, --method             (gauss"
#ifdef KEMFIELD_USE_PETSC
                        ", robinhood or PETSc)\n"
#else
                        " or robinhood)\n"
#endif
                        "\t -b, --integrator_type         (integrator_type 0=analytic, 1=RWG, 2=numeric)\n";

    int verbose = 3;
    int scale = 1;
    int power = 1;
    double accuracy = 1.e-8;
    (void) accuracy;
    int increment = 100;
#ifdef KEMFIELD_USE_VTK
    bool usePlot = false;
#endif
    int method = 1;
    int integrator_type = 2;

    static struct option longOptions[] = {
        {"help", no_argument, nullptr, 'h'},
        {"verbose", required_argument, nullptr, 'v'},
        {"scale", required_argument, nullptr, 's'},
        {"power", required_argument, nullptr, 'p'},
        {"accuracy", required_argument, nullptr, 'a'},
        {"increment", required_argument, nullptr, 'i'},
#ifdef KEMFIELD_USE_VTK
        {"with-plot", no_argument, nullptr, 'e'},
#endif
        {"method", required_argument, nullptr, 'm'},
        {"integrator_type", required_argument, nullptr, 'b'},
    };

#ifdef KEMFIELD_USE_VTK
    static const char* optString = "hv:s:p:t:a:i:em:b:";
#else
    static const char* optString = "hv:s:p:t:a:i:m:b:";
#endif

    while (true) {
        char optId = getopt_long(argc, argv, optString, longOptions, nullptr);
        if (optId == -1)
            break;
        switch (optId) {
            case ('h'):  // help
                MPI_SINGLE_PROCESS
                std::cout << usage << std::endl;
#ifdef KEMFIELD_USE_MPI
                KMPIInterface::GetInstance()->Finalize();
#endif
                return 0;
            case ('v'):  // verbose
                verbose = atoi(optarg);
                if (verbose < 0)
                    verbose = 0;
                if (verbose > 5)
                    verbose = 5;
                break;
            case ('s'):
                scale = atoi(optarg);
                break;
            case ('p'):
                power = atoi(optarg);
                break;
            case ('a'):
                accuracy = atof(optarg);
                break;
            case ('i'):
                increment = atoi(optarg);
                break;
#ifdef KEMFIELD_USE_VTK
            case ('e'):
                usePlot = true;
                break;
#endif
            case ('m'):
                method = atoi(optarg);
                break;
            case ('b'):
                integrator_type = atoi(optarg);
                break;
            default:  // unrecognized option
                MPI_SINGLE_PROCESS
                std::cout << usage << std::endl;
#ifdef KEMFIELD_USE_MPI
                KMPIInterface::GetInstance()->Finalize();
#endif
                return 1;
        }
    }

    if (scale < 1) {
        MPI_SINGLE_PROCESS
        std::cout << usage << std::endl;
#ifdef KEMFIELD_USE_MPI
        KMPIInterface::GetInstance()->Finalize();
#endif
        return 1;
    }

    KEMField::cout.Verbose(false);

    MPI_SINGLE_PROCESS
    KEMField::cout.Verbose(verbose != 0);

#if defined(KEMFIELD_USE_MPI) && defined(KEMFIELD_USE_OPENCL)
    KOpenCLInterface::GetInstance()->SetGPU(KMPIInterface::GetInstance()->GetProcess());
#endif

    KSurfaceContainer surfaceContainer;

    std::vector<double> segments(2 * scale);
    DiscretizeInterval(2 * 1., 2 * scale, power, segments);

    typedef KSurface<KElectrostaticBasis, KDirichletBoundary, KConicSection> KEMConicSection;

    double r0 = 1.;
    double r1 = 1.;
    for (int i = 0; i < scale; i++) {
        r1 -= segments.at(i);

        auto* cs = new KEMConicSection();
        cs->SetR0(r0);
        cs->SetZ0(0.);
        cs->SetR1(r1);
        cs->SetZ1(0.);
        cs->SetBoundaryValue(1.);

        surfaceContainer.push_back(cs);

        r0 = r1;
    }

    IntegratorOption integratorSelection = integratorOptionList.at(integrator_type);
    MPI_SINGLE_PROCESS
    {
        KEMField::cout << "Using " << integratorSelection.name << " integrator." << KEMField::endl;
    }

    if (method == 0)  // Gaussian Elimination
    {

        KElectrostaticBoundaryIntegrator integrator{integratorSelection.Create()};
        KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A(surfaceContainer, integrator);
        KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x(surfaceContainer, integrator);
        KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b(surfaceContainer, integrator);

        KGaussianElimination<KElectrostaticBoundaryIntegrator::ValueType> gaussianElimination;
        gaussianElimination.Solve(A, x, b);
    }
    else if (method == 1)  // Robin Hood
    {
#ifdef KEMFIELD_USE_OPENCL
        KOpenCLSurfaceContainer oclSurfaceContainer(surfaceContainer);
        KOpenCLElectrostaticBoundaryIntegrator integrator{integratorSelection.CreateOCL(oclSurfaceContainer)};
        KBoundaryIntegralMatrix<KOpenCLBoundaryIntegrator<KElectrostaticBasis>> A(oclSurfaceContainer, integrator);
        KBoundaryIntegralVector<KOpenCLBoundaryIntegrator<KElectrostaticBasis>> b(oclSurfaceContainer, integrator);
        KBoundaryIntegralSolutionVector<KOpenCLBoundaryIntegrator<KElectrostaticBasis>> x(oclSurfaceContainer,
                                                                                          integrator);
#else
        KElectrostaticBoundaryIntegrator integrator{integratorSelection.Create()};
        KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A(surfaceContainer, integrator);
        KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x(surfaceContainer, integrator);
        KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b(surfaceContainer, integrator);
#endif

#if defined(KEMFIELD_USE_MPI) && defined(KEMFIELD_USE_OPENCL)
        KRobinHood<KElectrostaticBoundaryIntegrator::ValueType, KRobinHood_MPI_OpenCL> robinHood;
#ifndef KEMFIELD_USE_DOUBLE_PRECISION
        robinHood.SetTolerance((accuracy > 1.e-5 ? accuracy : 1.e-5));
#else
        robinHood.SetTolerance(accuracy);
#endif
#elif defined(KEMFIELD_USE_MPI)
        KRobinHood<KElectrostaticBoundaryIntegrator::ValueType, KRobinHood_MPI> robinHood;
#elif defined(KEMFIELD_USE_OPENCL)
        KRobinHood<KElectrostaticBoundaryIntegrator::ValueType, KRobinHood_OpenCL> robinHood;
#ifndef KEMFIELD_USE_DOUBLE_PRECISION
        robinHood.SetTolerance((accuracy > 1.e-5 ? accuracy : 1.e-5));
#else
        robinHood.SetTolerance(accuracy);
#endif
#else
        KRobinHood<KElectrostaticBoundaryIntegrator::ValueType> robinHood;
#endif

        robinHood.AddVisitor(new KIterationDisplay<KElectrostaticBoundaryIntegrator::ValueType>());

#ifdef KEMFIELD_USE_VTK
        MPI_SINGLE_PROCESS
        if (usePlot)
            robinHood.AddVisitor(new KVTKIterationPlotter<KElectrostaticBoundaryIntegrator::ValueType>(5));
#endif

        robinHood.SetResidualCheckInterval(increment);
        robinHood.Solve(A, x, b);
    }  // robin hood method

#ifdef KEMFIELD_USE_PETSC
    else if (method == 2)  // in this particular case the numeric boundary integrator has been implemented exclusively
    {
        KElectrostaticBoundaryIntegrator integrator;
        KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A(surfaceContainer, integrator);
        KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x(surfaceContainer, integrator);
        KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b(surfaceContainer, integrator);

        KPETScSolver<KElectrostaticBoundaryIntegrator::ValueType> petscSolver;
        petscSolver.SetTolerance(accuracy);
        petscSolver.Solve(A, x, b);
    }
#endif

    MPI_SINGLE_PROCESS
    {
        double Q = 0.;

        unsigned int i = 0;
        for (KSurfaceContainer::iterator it = surfaceContainer.begin(); it != surfaceContainer.end(); it++) {
            Q += (static_cast<KEMConicSection*>(*it)->Area() * static_cast<KEMConicSection*>(*it)->GetSolution());
            i++;
        }

        std::cout << "" << std::endl;
        double C = Q / KEMConstants::Eps0;
        double C_analytic = 8.;

        std::cout << "Capacitance:    " << C << std::endl;
        std::cout << "Analytic value: " << C_analytic << std::endl;
        std::cout << "Error:          " << (fabs(C - C_analytic) / C_analytic) * 100 << " %" << std::endl;
    }

#ifdef KEMFIELD_USE_PETSC
    KPETScInterface::GetInstance()->Finalize();
#elif KEMFIELD_USE_MPI
    KMPIInterface::GetInstance()->Finalize();
#endif

    return 0;
}

void DiscretizeInterval(double interval, int nSegments, double power, std::vector<double>& segments)
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
            inc1 = ((double) i) / (nSegments / 2);
            inc2 = ((double) (i + 1)) / (nSegments / 2);

            inc1 = pow(inc1, power);
            inc2 = pow(inc2, power);

            segments[i] = segments[nSegments - (i + 1)] = mid * (inc2 - inc1);
        }
    }
}
