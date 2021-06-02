#include "KElectrostaticBoundaryIntegratorOptions.hh"
#include "KBinaryDataStreamer.hh"
#include "KBoundaryIntegralMatrix.hh"
#include "KBoundaryIntegralSolutionVector.hh"
#include "KBoundaryIntegralVector.hh"
#include "KDataDisplay.hh"
#include "KEMConstants.hh"
#include "KEMCout.hh"
#include "KGBEM.hh"
#include "KGBEMConverter.hh"
#include "KGBox.hh"
#include "KGMesher.hh"
#include "KGRectangle.hh"
#include "KGaussianElimination.hh"
#include "KIterationTracker.hh"
#include "KIterativeStateWriter.hh"
#include "KRobinHood.hh"
#include "KSADataStreamer.hh"
#include "KSurface.hh"
#include "KSurfaceContainer.hh"
#include "KSurfaceTypes.hh"
#include "KTypelist.hh"

#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef KEMFIELD_USE_MPI
#include "KRobinHood_MPI.hh"
//#define MPI_SINGLE_PROCESS if (KMPIInterface::GetInstance()->GetProcess()==0)
#else
//#define MPI_SINGLE_PROCESS
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

#include "KEMFieldCanvas.hh"

#ifdef KEMFIELD_USE_VTK
#include "KEMVTKFieldCanvas.hh"
#include "KEMVTKViewer.hh"
#include "KVTKIterationPlotter.hh"
#endif

#ifndef DEFAULT_OUTPUT_DIR
#define DEFAULT_OUTPUT_DIR "."
#endif /* !DEFAULT_OUTPUT_DIR */

using namespace KGeoBag;
using namespace KEMField;

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
                        "the results from F. Read, J. Comp. Phys. 113 (1997).\n"
                        "\n"
                        "\tAvailable options:\n"
                        "\t -h, --help               (shows this message and exits)\n"
                        "\t -v, --verbose            (bool; sets the verbosity)\n"
                        "\t -s, --scale              (discretization scale)\n"
                        "\t -p, --power              (discretization power)\n"
                        "\t -a, --accuracy           (accuracy of charge density computation)\n"
                        "\t -i, --increment          (increment of accuracy check)\n"
                        "\t -j, --save-increment     (increment of state save)\n"
#ifdef KEMFIELD_USE_VTK
                        "\t -e, --with-plot          (dynamic plot of residual norm)\n"
#endif
                        "\t -m, --method             (gauss"
#ifdef KEMFIELD_USE_PETSC
                        ", robinhood or PETSc)\n"
                        "\n";
#else
                        " or robinhood)\n"
                        "\n"
#endif
    "\t -b, --integrator_type         (integrator_type 0=analytic, 1=RWG, 2=numeric)\n";

    bool verbose = true;
    int scale = 1;
    int power = 1;
    double accuracy = 1.e-8;
    (void) accuracy;
    unsigned int increment = 100;
    unsigned int saveIncrement = UINT_MAX;
    bool usePlot = false;
    int method = 1;
    int integrator_type = 2;

    static struct option longOptions[] = {
        {"help", no_argument, nullptr, 'h'},
        {"verbose", required_argument, nullptr, 'v'},
        {"scale", required_argument, nullptr, 's'},
        {"power", required_argument, nullptr, 'p'},
        {"accuracy", required_argument, nullptr, 'a'},
        {"increment", required_argument, nullptr, 'i'},
        {"save-increment", required_argument, nullptr, 'j'},
#ifdef KEMFIELD_USE_VTK
        {"with-plot", no_argument, nullptr, 'e'},
#endif
        {"method", required_argument, nullptr, 'm'},
        {"integrator_type", required_argument, nullptr, 'b'},
    };

#ifdef KEMFIELD_USE_VTK
    static const char* optString = "hv:s:p:a:i:j:em:";
#else
    static const char* optString = "hv:s:p:a:i:j:m:";
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
                verbose = (atoi(optarg) != 0);
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
            case ('j'):
                saveIncrement = atoi(optarg);
                break;
            case ('e'):
                usePlot = true;
                break;
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
    KEMField::cout.Verbose(verbose);

#if defined(KEMFIELD_USE_MPI) && defined(KEMFIELD_USE_OPENCL)
    KOpenCLInterface::GetInstance()->SetGPU(KMPIInterface::GetInstance()->GetProcess() + 1);
#endif

    // Construct the shape
    auto* box = new KGBox();

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

    auto* cube = new KGSurface(box);
    cube->SetName("box");
    cube->MakeExtension<KGMesh>();
    cube->MakeExtension<KGElectrostaticDirichlet>();
    cube->AsExtension<KGElectrostaticDirichlet>()->SetBoundaryValue(1.);

    // Mesh the elements
    auto* mesher = new KGMesher();
    cube->AcceptNode(mesher);

    KSurfaceContainer surfaceContainer;
    KGBEMMeshConverter geometryConverter(surfaceContainer);
    cube->AcceptNode(&geometryConverter);

    MPI_SINGLE_PROCESS
    {
        KEMField::cout << "Computing the capacitance for a unit cube comprised of " << surfaceContainer.size()
                       << " elements" << KEMField::endl;
    }

    IntegratorOption integratorSelection = integratorOptionList.at(integrator_type);
    MPI_SINGLE_PROCESS
    {
        KEMField::cout << "Using " << integratorSelection.name << " integrator." << KEMField::endl;
    }

    if (method == 0) {
        KElectrostaticBoundaryIntegrator integrator{integratorSelection.Create()};
        KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A(surfaceContainer, integrator);
        KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x(surfaceContainer, integrator);
        KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b(surfaceContainer, integrator);

        KGaussianElimination<KElectrostaticBoundaryIntegrator::ValueType> gaussianElimination;
        gaussianElimination.Solve(A, x, b);
    }
    else if (method == 1)  // robin hood
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

        MPI_SINGLE_PROCESS
        {
            auto* tracker = new KIterationTracker<KElectrostaticBoundaryIntegrator::ValueType>();
            tracker->Interval(1);
            tracker->WriteInterval(5);
            tracker->MaxIterationStamps(10);
            robinHood.AddVisitor(tracker);
        }

        auto* stateWriter = new KIterativeStateWriter<KElectrostaticBoundaryIntegrator::ValueType>(surfaceContainer);
        stateWriter->Interval(saveIncrement);
        robinHood.AddVisitor(stateWriter);

#ifdef KEMFIELD_USE_VTK
        MPI_SINGLE_PROCESS
        if (usePlot)
            robinHood.AddVisitor(new KVTKIterationPlotter<KElectrostaticBoundaryIntegrator::ValueType>(5));
#endif

        robinHood.SetResidualCheckInterval(increment);
        robinHood.Solve(A, x, b);
    }  // robin hood

#ifdef KEMFIELD_USE_PETSC
    else if (method == 2) {
        KElectrostaticBoundaryIntegrator integrator{integratorSelection.Create()};
        KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A(surfaceContainer, integrator);
        KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x(surfaceContainer, integrator);
        KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b(surfaceContainer, integrator);

        KPETScSolver<KElectrostaticBoundaryIntegrator::ValueType> petscSolver;
        petscSolver.SetTolerance(accuracy);
        petscSolver.Solve(A, x, b);
    }
#endif

    if (usePlot) {
        KElectrostaticBoundaryIntegrator integrator{integratorSelection.Create()};
        KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A(surfaceContainer, integrator);

        KEMFieldCanvas* fieldCanvas = nullptr;

#if defined(KEMFIELD_USE_VTK)
        fieldCanvas = new KEMVTKFieldCanvas(0., double(A.Dimension()), 0., double(A.Dimension()), 1.e30, true);
#endif

        if (fieldCanvas != nullptr) {
            std::vector<double> x_;
            std::vector<double> y_;
            std::vector<double> V_;

            for (unsigned int i = 0; i < A.Dimension(); i++) {
                x_.push_back(i);
                y_.push_back(i);

                for (unsigned int j = 0; j < A.Dimension(); j++) {
                    double value = A(i, j);
                    if (value > 1.e-16)
                        V_.push_back(log(value));
                    else
                        V_.push_back(-16.);
                }
            }

            fieldCanvas->DrawFieldMap(x_, y_, V_, false, 0.);
            fieldCanvas->LabelAxes("i", "j", "log (A_{ij})");
            std::string fieldCanvasName = DEFAULT_OUTPUT_DIR;
            fieldCanvas->SaveAs(fieldCanvasName + "/Matrix.gif");
        }
    }

    MPI_SINGLE_PROCESS
    {
        double Q = 0.;

        unsigned int i = 0;
        for (KSurfaceContainer::iterator it = surfaceContainer.begin(); it != surfaceContainer.end(); it++) {
            Q += (dynamic_cast<KRectangle*>(*it)->Area() * dynamic_cast<KElectrostaticBasis*>(*it)->GetSolution());
            i++;
        }

        std::cout << "" << std::endl;
        double C = Q / (4. * M_PI * KEMConstants::Eps0);

        double C_Read = 0.6606785;
        double C_Read_err = 0.0000006;

        std::cout << std::setprecision(7) << "Capacitance:    " << C << std::endl;
        std::cout.setf(std::ios::fixed, std::ios::floatfield);
        std::cout << std::setprecision(7) << "Accepted value: " << C_Read << " +\\- " << C_Read_err << std::endl;
        std::cout << "Accuracy:       " << (fabs(C - C_Read) / C_Read) * 100 << " %" << std::endl;
    }

#ifdef KEMFIELD_USE_VTK
    MPI_SINGLE_PROCESS
    {
        KEMVTKViewer viewer(surfaceContainer);
        viewer.GenerateGeometryFile("cube.vtp");
        // viewer.ViewGeometry();
    }
#endif

#ifdef KEMFIELD_USE_PETSC
    KPETScInterface::GetInstance()->Finalize();
#elif KEMFIELD_USE_MPI
    KMPIInterface::GetInstance()->Finalize();
#endif

    return 0;
}
