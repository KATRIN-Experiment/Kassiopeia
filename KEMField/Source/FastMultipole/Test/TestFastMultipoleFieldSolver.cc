#include "KBoundaryIntegralMatrix.hh"
#include "KBoundaryIntegralSolutionVector.hh"
#include "KBoundaryIntegralVector.hh"
#include "KEMFileInterface.hh"
#include "KGBEM.hh"
#include "KGBEMConverter.hh"
#include "KGBox.hh"
#include "KGMesher.hh"
#include "KGRectangle.hh"
#include "KGRotatedObject.hh"
#include "KSurface.hh"
#include "KSurfaceContainer.hh"
#include "KSurfaceTypes.hh"
#include "KThreeVector_KEMField.hh"
#include "KTypelist.hh"

#include <cstdlib>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef KEMFIELD_USE_OPENCL
#include "KOpenCLBoundaryIntegralMatrix.hh"
#include "KOpenCLBoundaryIntegralSolutionVector.hh"
#include "KOpenCLBoundaryIntegralVector.hh"
#include "KOpenCLElectrostaticBoundaryIntegratorFactory.hh"
#include "KOpenCLElectrostaticIntegratingFieldSolver.hh"
#include "KOpenCLSurfaceContainer.hh"
#endif

#include "KElectrostaticBoundaryIntegratorFactory.hh"
#include "KElectrostaticIntegratingFieldSolver.hh"
#include "KFMElectrostaticFastMultipoleBoundaryValueSolver.hh"
#include "KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration.hh"
#include "KFMElectrostaticFastMultipoleFieldSolver.hh"
#include "KFMElectrostaticFieldMapper_SingleThread.hh"
#include "KFMElectrostaticParameters.hh"
#include "KFMElectrostaticParametersConfiguration.hh"
#include "KFMElectrostaticTree.hh"
#include "KFMElectrostaticTreeConstructor.hh"
#ifdef KEMFIELD_USE_OPENCL
#include "KFMElectrostaticFastMultipoleFieldSolver_OpenCL.hh"
#include "KFMElectrostaticFieldMapper_OpenCL.hh"
#endif

#include "KFMNamedScalarData.hh"
#include "KFMNamedScalarDataCollection.hh"


#ifdef KEMFIELD_USE_ROOT
#include "TApplication.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TGraph.h"
#include "TGraph2D.h"
#include "TStyle.h"
#endif

#include "KMPIEnvironment.hh"

using namespace KGeoBag;
using namespace KEMField;
using katrin::KThreeVector;

int main(int argc, char** argv)
{
#if KEMFIELD_USE_MPI
    //if you wish to exploit task based parallelism when using mpi
    //recompile this with use_split = true
    //this separates the sparse and dense matrix vector products into
    //separate processes, hoever if this is done then an even number of processes
    //must be launched
    bool use_split = false;
    KMPIInterface::GetInstance()->Initialize(&argc, &argv, use_split);
#endif

    std::string usage =
        "\n"
        "Usage: TestFastMultipoleFieldSolver <config file> <options>\n"
        "\n"
        "This program computes the solution of a simple dirichlet problem and compares the fast multipole field to the direct sovler. \n"
        "\tAvailable options:\n"
        "\t -h, --help               (shows this message and exits)\n"
        "\t -c  --config             (full path to configuration file)\n"
        "\t -s, --scale              (scale of geometry discretization)\n"
        "\t -g, --geometry           (0: Single sphere, \n"
        "\t                           1: Single cube, \n"
        "\t                           2: Cube and sphere \n"
        "\t                           3: Spherical capacitor \n"
        "\t -n, --n-evaluations      (number of evaluations in field map along each dimension) \n"
        "\t -m, --mode               (0: evaluate points along x axis \n"
        "\t                           1: evaluate points along y axis \n"
        "\t                           2: evaluate points along z axis \n"
        "\t                           3: evaluate points in y-z plane \n"
        "\t                           4: evaluate points in x-z plane \n"
        "\t                           5: evaluate points in x-y plane \n";

    std::string config_file;
    unsigned int scale = 20;
    unsigned int geometry = 0;
    unsigned int n_evaluations = 50;
    unsigned int mode = 2;

    static struct option longOptions[] = {{"help", no_argument, nullptr, 'h'},
                                          {"config", required_argument, nullptr, 'c'},
                                          {"scale", required_argument, nullptr, 's'},
                                          {"geometry", required_argument, nullptr, 'g'},
                                          {"n-evaluations", required_argument, nullptr, 'n'},
                                          {"mode", required_argument, nullptr, 'm'}};

    static const char* optString = "hc:s:g:n:m:";

    while (true) {
        char optId = getopt_long(argc, argv, optString, longOptions, nullptr);
        if (optId == -1)
            break;
        switch (optId) {
            case ('h'):  // help
                MPI_SINGLE_PROCESS
                {
                    std::cout << usage << std::endl;
                }
#ifdef KEMFIELD_USE_MPI
                KMPIInterface::GetInstance()->Finalize();
#endif
                return 0;
            case ('c'):
                config_file = std::string(optarg);
                break;
            case ('s'):
                scale = atoi(optarg);
                break;
            case ('g'):
                geometry = atoi(optarg);
                break;
            case ('n'):
                n_evaluations = atoi(optarg);
                break;
            case ('m'):
                mode = atoi(optarg);
                break;
            default:
                MPI_SINGLE_PROCESS
                {
                    std::cout << usage << std::endl;
                }
#ifdef KEMFIELD_USE_MPI
                KMPIInterface::GetInstance()->Finalize();
#endif
                return 1;
        }
    }

    //need to read in tree parameters
    bool solver_parameters_present = false;
    auto* solver_parameter_input =
        new KSAObjectInputNode<KFMElectrostaticParametersConfiguration>(std::string("SolverElectrostaticParameters"));
    KEMFileInterface::GetInstance()->ReadKSAFile(solver_parameter_input, config_file, solver_parameters_present);

    //need to read in boundary value problem solver parameters
    bool bvp_parameters_present = false;
    auto* bvp_config_input = new KSAObjectInputNode<KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration>(
        std::string("BoundaryValueSolverConfiguration"));
    KEMFileInterface::GetInstance()->ReadKSAFile(bvp_config_input, config_file, bvp_parameters_present);

    //optionally present in config file
    bool preconditioner_parameters_present = false;
    auto* preconditioner_parameter_input = new KSAObjectInputNode<KFMElectrostaticParametersConfiguration>(
        std::string("PreconditionerElectrostaticParameters"));
    KEMFileInterface::GetInstance()->ReadKSAFile(preconditioner_parameter_input,
                                                 config_file,
                                                 preconditioner_parameters_present);

    if (solver_parameters_present && bvp_parameters_present) {
        MPI_SINGLE_PROCESS
        {
            kfmout << "Configuration file: " << config_file << " read successfully." << kfmendl;
        }
    }
    else {
        MPI_SINGLE_PROCESS
        {
            kfmout << "Configuration file: " << config_file << " read failed." << kfmendl;
        }
#ifdef KEMFIELD_USE_MPI
        KMPIInterface::GetInstance()->Finalize();
#endif
        return 1;
    }


    KSurfaceContainer surfaceContainer;

    if (geometry == 0 || geometry == 2) {

        // Construct the shape
        double p1[2], p2[2];
        double radius = 1.;
        auto* hemi1 = new KGRotatedObject(scale, 20);
        p1[0] = -1.;
        p1[1] = 0.;
        p2[0] = 0.;
        p2[1] = 1.;
        hemi1->AddArc(p2, p1, radius, true);

        auto* hemi2 = new KGRotatedObject(scale, 20);
        p2[0] = 1.;
        p2[1] = 0.;
        p1[0] = 0.;
        p1[1] = 1.;
        hemi2->AddArc(p1, p2, radius, false);

        // Construct shape placement
        auto* h1 = new KGRotatedSurface(hemi1);
        auto* hemisphere1 = new KGSurface(h1);
        hemisphere1->SetName("hemisphere1");
        hemisphere1->MakeExtension<KGMesh>();
        hemisphere1->MakeExtension<KGElectrostaticDirichlet>();
        hemisphere1->AsExtension<KGElectrostaticDirichlet>()->SetBoundaryValue(-1.);

        auto* h2 = new KGRotatedSurface(hemi2);
        auto* hemisphere2 = new KGSurface(h2);
        hemisphere2->SetName("hemisphere2");
        hemisphere2->MakeExtension<KGMesh>();
        hemisphere2->MakeExtension<KGElectrostaticDirichlet>();
        hemisphere2->AsExtension<KGElectrostaticDirichlet>()->SetBoundaryValue(-1.);

        // Mesh the elements
        auto* mesher = new KGMesher();
        hemisphere1->AcceptNode(mesher);
        hemisphere2->AcceptNode(mesher);

        KGBEMMeshConverter geometryConverter(surfaceContainer);
        geometryConverter.SetMinimumArea(1.e-12);
        hemisphere1->AcceptNode(&geometryConverter);
        hemisphere2->AcceptNode(&geometryConverter);
    }

    if (geometry == 1 || geometry == 2) {

        // Construct the shape
        auto* box = new KGBox();
        int meshCount = scale;

        box->SetX0(-.5);
        box->SetX1(.5);
        box->SetXMeshCount(meshCount);
        box->SetXMeshPower(3);

        box->SetY0(-.5);
        box->SetY1(.5);
        box->SetYMeshCount(meshCount);
        box->SetYMeshPower(3);

        box->SetZ0(-.5);
        box->SetZ1(.5);
        box->SetZMeshCount(meshCount);
        box->SetZMeshPower(3);

        auto* cube = new KGSurface(box);
        cube->SetName("box");
        cube->MakeExtension<KGMesh>();
        cube->MakeExtension<KGElectrostaticDirichlet>();
        cube->AsExtension<KGElectrostaticDirichlet>()->SetBoundaryValue(1.0);

        // Mesh the elements
        auto* mesher = new KGMesher();
        cube->AcceptNode(mesher);

        KGBEMMeshConverter geometryConverter(surfaceContainer);
        cube->AcceptNode(&geometryConverter);
    }


    if (geometry == 3) {

        double radius1 = 1.;
        double radius2 = 2.;
        double radius3 = 3.;

        double potential1 = 1.;
        double potential2;
        potential2 = 0.;
        double permittivity1 = 2.;
        double permittivity2 = 3.;


        // Construct the shapes
        double p1[2], p2[2];
        double radius = radius1;

        auto* innerhemi1 = new KGRotatedObject(scale * 10, 10);
        p1[0] = -radius;
        p1[1] = 0.;
        p2[0] = 0.;
        p2[1] = radius;
        innerhemi1->AddArc(p2, p1, radius, true);

        auto* innerhemi2 = new KGRotatedObject(scale * 10, 10);
        p2[0] = radius;
        p2[1] = 0.;
        p1[0] = 0.;
        p1[1] = radius;
        innerhemi2->AddArc(p1, p2, radius, false);

        radius = radius2;

        auto* middlehemi1 = new KGRotatedObject(20 * scale, 10);
        p1[0] = -radius;
        p1[1] = 0.;
        p2[0] = 0.;
        p2[1] = radius;
        middlehemi1->AddArc(p2, p1, radius, true);

        auto* middlehemi2 = new KGRotatedObject(20 * scale, 10);
        p2[0] = radius;
        p2[1] = 0.;
        p1[0] = 0.;
        p1[1] = radius;
        middlehemi2->AddArc(p1, p2, radius, false);

        radius = radius3;

        auto* outerhemi1 = new KGRotatedObject(30 * scale, 10);
        p1[0] = -radius;
        p1[1] = 0.;
        p2[0] = 0.;
        p2[1] = radius;
        outerhemi1->AddArc(p2, p1, radius, true);

        auto* outerhemi2 = new KGRotatedObject(30 * scale, 10);
        p2[0] = radius;
        p2[1] = 0.;
        p1[0] = 0.;
        p1[1] = radius;
        outerhemi2->AddArc(p1, p2, radius, false);

        // Construct shape placement
        auto* ih1 = new KGRotatedSurface(innerhemi1);
        auto* innerhemisphere1 = new KGSurface(ih1);
        innerhemisphere1->SetName("innerhemisphere1");
        innerhemisphere1->MakeExtension<KGMesh>();
        innerhemisphere1->MakeExtension<KGElectrostaticDirichlet>()->SetBoundaryValue(potential1);

        auto* ih2 = new KGRotatedSurface(innerhemi2);
        auto* innerhemisphere2 = new KGSurface(ih2);
        innerhemisphere2->SetName("innerhemisphere2");
        innerhemisphere2->MakeExtension<KGMesh>();
        innerhemisphere2->MakeExtension<KGElectrostaticDirichlet>()->SetBoundaryValue(potential1);

        auto* mh1 = new KGRotatedSurface(middlehemi1);
        auto* middlehemisphere1 = new KGSurface(mh1);
        middlehemisphere1->SetName("middlehemisphere1");
        middlehemisphere1->MakeExtension<KGMesh>();
        middlehemisphere1->MakeExtension<KGElectrostaticNeumann>()->SetNormalBoundaryFlux(permittivity2 /
                                                                                          permittivity1);

        auto* mh2 = new KGRotatedSurface(middlehemi2);
        auto* middlehemisphere2 = new KGSurface(mh2);
        middlehemisphere2->SetName("middlehemisphere2");
        middlehemisphere2->MakeExtension<KGMesh>();
        middlehemisphere2->MakeExtension<KGElectrostaticNeumann>()->SetNormalBoundaryFlux(permittivity1 /
                                                                                          permittivity2);

        auto* oh1 = new KGRotatedSurface(outerhemi1);
        auto* outerhemisphere1 = new KGSurface(oh1);
        outerhemisphere1->SetName("outerhemisphere1");
        outerhemisphere1->MakeExtension<KGMesh>();
        outerhemisphere1->MakeExtension<KGElectrostaticDirichlet>()->SetBoundaryValue(potential2);
        auto* oh2 = new KGRotatedSurface(outerhemi2);
        auto* outerhemisphere2 = new KGSurface(oh2);
        outerhemisphere2->SetName("outerhemisphere2");
        outerhemisphere2->MakeExtension<KGMesh>();
        outerhemisphere2->MakeExtension<KGElectrostaticDirichlet>()->SetBoundaryValue(potential2);

        // Mesh the elements
        auto* mesher = new KGMesher();
        innerhemisphere1->AcceptNode(mesher);
        innerhemisphere2->AcceptNode(mesher);
        middlehemisphere1->AcceptNode(mesher);
        middlehemisphere2->AcceptNode(mesher);
        outerhemisphere1->AcceptNode(mesher);
        outerhemisphere2->AcceptNode(mesher);

        KGBEMMeshConverter geometryConverter(surfaceContainer);
        geometryConverter.SetMinimumArea(1.e-12);
        innerhemisphere1->AcceptNode(&geometryConverter);
        innerhemisphere2->AcceptNode(&geometryConverter);
        middlehemisphere1->AcceptNode(&geometryConverter);
        middlehemisphere2->AcceptNode(&geometryConverter);
        outerhemisphere1->AcceptNode(&geometryConverter);
        outerhemisphere2->AcceptNode(&geometryConverter);
    }

    MPI_SINGLE_PROCESS
    {
        KEMField::cout << "Number of elements in BEM problem = " << surfaceContainer.size() << KEMField::endl;
    }

    ////////////////////////////////////////////////////////////////////////////////

    //get the tree fast multipole parameters
    KFMElectrostaticParameters solver_params = solver_parameter_input->GetObject()->GetParameters();

    //get the tree fast multipole parameters
    KFMElectrostaticParameters precon_params;
    if (preconditioner_parameters_present) {
        precon_params = preconditioner_parameter_input->GetObject()->GetParameters();
    }

    auto* solver = new KFMElectrostaticFastMultipoleBoundaryValueSolver();
    solver->SetConfigurationObject(bvp_config_input->GetObject());
    solver->SetSolverElectrostaticParameters(solver_params);
    //will not be used unless the preconditioner is the independent_implicit_krylov type
    solver->SetPreconditionerElectrostaticParameters(precon_params);

    //solve the boundary value problem
    solver->Solve(surfaceContainer);

    delete solver;

    MPI_SINGLE_PROCESS
    {
        std::cout << "Done charge density solving." << std::endl;
    }

    if (geometry == 1) {
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
        MPI_SINGLE_PROCESS
        {
            std::cout << std::setprecision(7) << "Capacitance:    " << C << std::endl;
            std::cout.setf(std::ios::fixed, std::ios::floatfield);
            std::cout << std::setprecision(7) << "Accepted value: " << C_Read << " +\\- " << C_Read_err << std::endl;
            std::cout << "Accuracy:       " << (fabs(C - C_Read) / C_Read) * 100 << " %" << std::endl;
        }
    }


    if (geometry == 0) {
        double Q = 0.;
        unsigned int i = 0;
        for (KSurfaceContainer::iterator it = surfaceContainer.begin(); it != surfaceContainer.end(); it++) {
            Q += (dynamic_cast<KTriangle*>(*it)->Area() * dynamic_cast<KElectrostaticBasis*>(*it)->GetSolution());
            i++;
        }
        std::cout << "" << std::endl;
        double C = Q / (4. * M_PI * KEMConstants::Eps0);
        double C_Read = -1.0;
        double C_Read_err = 0.000000;
        MPI_SINGLE_PROCESS
        {
            std::cout << std::setprecision(7) << "Capacitance:    " << C << std::endl;
            std::cout.setf(std::ios::fixed, std::ios::floatfield);
            std::cout << std::setprecision(7) << "Accepted value: " << C_Read << " +\\- " << C_Read_err << std::endl;
            std::cout << "Accuracy:       " << (fabs(C - C_Read) / C_Read) * 100 << " %" << std::endl;
        }
    }


    (void) mode;
    (void) n_evaluations;

    MPI_SINGLE_PROCESS
    {

        KFMElectrostaticParameters params = solver_params;

        params.region_expansion_factor = 2.1;
        //now build the field solvers

#ifdef KEMFIELD_USE_OPENCL
        //re-create opencl container
        KOpenCLSurfaceContainer* oclSurfaceContainer;
        oclSurfaceContainer = NULL;
        KOpenCLInterface::GetInstance()->SetActiveData(oclSurfaceContainer);  //set to NULL

        //rebuild the ocl container
        KOpenCLData* ocl_data = KOpenCLInterface::GetInstance()->GetActiveData();
        if (ocl_data) {
            oclSurfaceContainer = dynamic_cast<KOpenCLSurfaceContainer*>(ocl_data);
        }

        if (oclSurfaceContainer == NULL) {
            oclSurfaceContainer = new KOpenCLSurfaceContainer(surfaceContainer);
            KOpenCLInterface::GetInstance()->SetActiveData(oclSurfaceContainer);
        }

        KOpenCLElectrostaticBoundaryIntegrator direct_integrator{KoclEBIFactory::MakeDefault(*oclSurfaceContainer)};
        KIntegratingFieldSolver<KOpenCLElectrostaticBoundaryIntegrator>* direct_solver =
            new KIntegratingFieldSolver<KOpenCLElectrostaticBoundaryIntegrator>(*oclSurfaceContainer,
                                                                                direct_integrator);
        direct_solver->Initialize();
#else
        KElectrostaticBoundaryIntegrator direct_integrator_single_thread{KEBIFactory::MakeDefault()};
        auto* direct_solver =
            new KIntegratingFieldSolver<KElectrostaticBoundaryIntegrator>(surfaceContainer,
                                                                          direct_integrator_single_thread);
#endif

        //create a tree
        auto* tree = new KFMElectrostaticTree();
        tree->SetParameters(params);

#ifdef KEMFIELD_USE_OPENCL
        KFMElectrostaticTreeConstructor<KFMElectrostaticFieldMapper_OpenCL> constructor;
        //    constructor.ConstructTree(surfaceContainer, *tree);
        constructor.ConstructTree(*oclSurfaceContainer, *tree);
#else
        KFMElectrostaticTreeConstructor<KFMElectrostaticFieldMapper_SingleThread> constructor;
        constructor.ConstructTree(surfaceContainer, *tree);
#endif

//now build the fast multipole field solver
#ifdef KEMFIELD_USE_OPENCL
        KFMElectrostaticFastMultipoleFieldSolver_OpenCL* fast_solver =
            new KFMElectrostaticFastMultipoleFieldSolver_OpenCL(KoclEBIFactory::MakeDefaultConfig(),
                                                                *oclSurfaceContainer,
                                                                *tree);
#else
        auto* fast_solver = new KFMElectrostaticFastMultipoleFieldSolver(surfaceContainer, *tree);
#endif

        std::cout << "starting field evaluation" << std::endl;


        ////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////

        KFMNamedScalarData x_coord;
        x_coord.SetName("x_coordinate");
        KFMNamedScalarData y_coord;
        y_coord.SetName("y_coordinate");
        KFMNamedScalarData z_coord;
        z_coord.SetName("z_coordinate");
        KFMNamedScalarData fmm_potential;
        fmm_potential.SetName("fast_multipole_potential");
        KFMNamedScalarData direct_potential;
        direct_potential.SetName("direct_potential");
        KFMNamedScalarData potential_error;
        potential_error.SetName("potential_error");
        KFMNamedScalarData log_potential_error;
        log_potential_error.SetName("log_potential_error");

        KFMNamedScalarData fmm_field_x;
        fmm_field_x.SetName("fast_multipole_field_x");
        KFMNamedScalarData fmm_field_y;
        fmm_field_y.SetName("fast_multipole_field_y");
        KFMNamedScalarData fmm_field_z;
        fmm_field_z.SetName("fast_multipole_field_z");

        KFMNamedScalarData direct_field_x;
        direct_field_x.SetName("direct_field_x");
        KFMNamedScalarData direct_field_y;
        direct_field_y.SetName("direct_field_y");
        KFMNamedScalarData direct_field_z;
        direct_field_z.SetName("direct_field_z");

        KFMNamedScalarData field_error_x;
        field_error_x.SetName("field_error_x");
        KFMNamedScalarData field_error_y;
        field_error_y.SetName("field_error_y");
        KFMNamedScalarData field_error_z;
        field_error_z.SetName("field_error_z");

        KFMNamedScalarData l2_field_error;
        l2_field_error.SetName("l2_field_error");
        KFMNamedScalarData logl2_field_error;
        logl2_field_error.SetName("logl2_field_error");

        KFMNamedScalarData fmm_time_per_potential_call;
        fmm_time_per_potential_call.SetName("fmm_time_per_potential_call");
        KFMNamedScalarData fmm_time_per_field_call;
        fmm_time_per_field_call.SetName("fmm_time_per_field_call");
        KFMNamedScalarData direct_time_per_potential_call;
        direct_time_per_potential_call.SetName("direct_time_per_potential_call");
        KFMNamedScalarData direct_time_per_field_call;
        direct_time_per_field_call.SetName("direct_time_per_field_call");

        //compute the positions of the evaluation points
        KFMCube<3>* world =
            KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCube<3>>::GetNodeObject(tree->GetRootNode());
        double world_length = world->GetLength();
        double length_a = world_length / 2.0 - 0.001 * world_length;
        double length_b = world_length / 2.0 - 0.001 * world_length;
        KFMPoint<3> center = world->GetCenter();

        KThreeVector direction_a;
        KThreeVector direction_b;
        KThreeVector direction_c;

        KThreeVector p0(center[0], center[1], center[2]);
        KThreeVector point;

        unsigned int n_points = 0;
        KThreeVector* points = nullptr;

        switch (mode) {
            case 0:  //x-axis
                n_points = n_evaluations;
                points = new KThreeVector[n_points];
                direction_a[0] = 1.0;
                direction_a[1] = 0.0;
                direction_a[2] = 0.0;
                p0 = p0 - length_a * direction_a;
                for (unsigned int i = 0; i < n_evaluations; i++) {
                    point = p0 + i * (2.0 * length_a / n_evaluations) * direction_a;
                    x_coord.AddNextValue(point[0]);
                    y_coord.AddNextValue(point[1]);
                    z_coord.AddNextValue(point[2]);
                    points[i] = point;
                }
                break;
            case 1:  //y-axis
                n_points = n_evaluations;
                points = new KThreeVector[n_points];
                direction_a[0] = 0.0;
                direction_a[1] = 1.0;
                direction_a[2] = 0.0;
                p0 = p0 - length_a * direction_a;
                for (unsigned int i = 0; i < n_evaluations; i++) {
                    point = p0 + i * (2.0 * length_a / n_evaluations) * direction_a;
                    x_coord.AddNextValue(point[0]);
                    y_coord.AddNextValue(point[1]);
                    z_coord.AddNextValue(point[2]);
                    points[i] = point;
                }
                break;
            case 2:  //z-axis
                n_points = n_evaluations;
                points = new KThreeVector[n_points];
                direction_a[0] = 0.0;
                direction_a[1] = 0.0;
                direction_a[2] = 1.0;
                p0 = p0 - length_a * direction_a;
                for (unsigned int i = 0; i < n_evaluations; i++) {
                    point = p0 + i * (2.0 * length_a / n_evaluations) * direction_a;
                    x_coord.AddNextValue(point[0]);
                    y_coord.AddNextValue(point[1]);
                    z_coord.AddNextValue(point[2]);
                    points[i] = point;
                }
                break;
            case 3:  //y-z plane
                n_points = n_evaluations * n_evaluations;
                points = new KThreeVector[n_points];
                direction_a[0] = 0.0;
                direction_a[1] = 1.0;
                direction_a[2] = 0.0;
                direction_b[0] = 0.0;
                direction_b[1] = 0.0;
                direction_b[2] = 1.0;
                p0 = p0 - length_a * direction_a;
                p0 = p0 - length_b * direction_b;
                for (unsigned int i = 0; i < n_evaluations; i++) {
                    for (unsigned int j = 0; j < n_evaluations; j++) {
                        point = p0 + i * (2.0 * length_a / n_evaluations) * direction_a +
                                j * (2.0 * length_b / n_evaluations) * direction_b;
                        x_coord.AddNextValue(point[0]);
                        y_coord.AddNextValue(point[1]);
                        z_coord.AddNextValue(point[2]);
                        points[j + n_evaluations * i] = point;
                    }
                }
                break;
            case 4:  //x-z plane
                n_points = n_evaluations * n_evaluations;
                points = new KThreeVector[n_points];
                direction_a[0] = 1.0;
                direction_a[1] = 0.0;
                direction_a[2] = 0.0;
                direction_b[0] = 0.0;
                direction_b[1] = 0.0;
                direction_b[2] = 1.0;
                p0 = p0 - length_a * direction_a;
                p0 = p0 - length_b * direction_b;
                for (unsigned int i = 0; i < n_evaluations; i++) {
                    for (unsigned int j = 0; j < n_evaluations; j++) {
                        point = p0 + i * (2.0 * length_a / n_evaluations) * direction_a +
                                j * (2.0 * length_b / n_evaluations) * direction_b;
                        x_coord.AddNextValue(point[0]);
                        y_coord.AddNextValue(point[1]);
                        z_coord.AddNextValue(point[2]);
                        points[j + n_evaluations * i] = point;
                    }
                }
                break;
            case 5:  //x-y plane
                n_points = n_evaluations * n_evaluations;
                points = new KThreeVector[n_points];
                direction_a[0] = 1.0;
                direction_a[1] = 0.0;
                direction_a[2] = 0.0;
                direction_b[0] = 0.0;
                direction_b[1] = 1.0;
                direction_b[2] = 0.0;
                p0 = p0 - length_a * direction_a;
                p0 = p0 - length_b * direction_b;
                for (unsigned int i = 0; i < n_evaluations; i++) {
                    for (unsigned int j = 0; j < n_evaluations; j++) {
                        point = p0 + i * (2.0 * length_a / n_evaluations) * direction_a +
                                j * (2.0 * length_b / n_evaluations) * direction_b;
                        x_coord.AddNextValue(point[0]);
                        y_coord.AddNextValue(point[1]);
                        z_coord.AddNextValue(point[2]);
                        points[j + n_evaluations * i] = point;
                    }
                }
                break;
        }


        std::cout << "n points = " << n_points << std::endl;
        std::cout << std::setprecision(7);


        //timer
        clock_t start, end;
        double time;

        start = clock();

        //evaluate multipole potential
        for (unsigned int i = 0; i < n_points; i++) {
            double potential = fast_solver->Potential(points[i]);
            std::cout << "potential = " << potential << std::endl;
            fmm_potential.AddNextValue(potential);
        }

        end = clock();
        time = ((double) (end - start)) / CLOCKS_PER_SEC;  // time in seconds
        time /= (double) (n_points);
        std::cout << "done fmm potential eval" << std::endl;
        std::cout << " time per fmm potential evaluation = " << time << std::endl;
        fmm_time_per_potential_call.AddNextValue(time);


        start = clock();

        //evaluate multipole field
        //evaluate multipole potential
        for (unsigned int i = 0; i < n_points; i++) {
            KThreeVector field = fast_solver->ElectricField(points[i]);
            fmm_field_x.AddNextValue(field[0]);
            fmm_field_y.AddNextValue(field[1]);
            fmm_field_z.AddNextValue(field[2]);
        }

        end = clock();
        time = ((double) (end - start)) / CLOCKS_PER_SEC;  // time in seconds
        time /= (double) (n_points);
        std::cout << "done fmm field eval" << std::endl;
        std::cout << " time per fmm field evaluation = " << time << std::endl;
        fmm_time_per_field_call.AddNextValue(time);


        start = clock();

        //evaluate direct potential
        for (unsigned int i = 0; i < n_points; i++) {
            direct_potential.AddNextValue(direct_solver->Potential(points[i]));
        }

        end = clock();
        time = ((double) (end - start)) / CLOCKS_PER_SEC;  // time in seconds
        time /= (double) (n_points);
        std::cout << "done direct potential eval" << std::endl;
        std::cout << " time per direct potential evaluation = " << time << std::endl;
        direct_time_per_potential_call.AddNextValue(time);


        start = clock();
        //evaluate direct field
        for (unsigned int i = 0; i < n_points; i++) {
            KThreeVector field = direct_solver->ElectricField(points[i]);
            direct_field_x.AddNextValue(field[0]);
            direct_field_y.AddNextValue(field[1]);
            direct_field_z.AddNextValue(field[2]);
        }

        end = clock();
        time = ((double) (end - start)) / CLOCKS_PER_SEC;  // time in seconds
        time /= (double) (n_points);
        std::cout << "done direct field eval" << std::endl;
        std::cout << " time per direct field evaluation = " << time << std::endl;
        direct_time_per_field_call.AddNextValue(time);


        //compute the errors
        for (unsigned int i = 0; i < n_points; i++) {
            unsigned int index = i;
            double pot_err = std::fabs(fmm_potential.GetValue(index) - direct_potential.GetValue(index));
            potential_error.AddNextValue(pot_err);
            log_potential_error.AddNextValue(std::max(std::log10(pot_err), -20.));

            double err_x = fmm_field_x.GetValue(index) - direct_field_x.GetValue(index);
            double err_y = fmm_field_y.GetValue(index) - direct_field_y.GetValue(index);
            double err_z = fmm_field_z.GetValue(index) - direct_field_z.GetValue(index);
            double l2_err = std::sqrt(err_x * err_x + err_y * err_y + err_z * err_z);

            field_error_x.AddNextValue(err_x);
            field_error_y.AddNextValue(err_y);
            field_error_z.AddNextValue(err_z);
            l2_field_error.AddNextValue(l2_err);
            logl2_field_error.AddNextValue(std::log10(l2_err));
        }

        KFMNamedScalarDataCollection data_collection;
        data_collection.AddData(x_coord);
        data_collection.AddData(y_coord);
        data_collection.AddData(z_coord);
        data_collection.AddData(fmm_potential);
        data_collection.AddData(direct_potential);
        data_collection.AddData(potential_error);
        data_collection.AddData(log_potential_error);
        data_collection.AddData(fmm_field_x);
        data_collection.AddData(fmm_field_y);
        data_collection.AddData(fmm_field_z);
        data_collection.AddData(direct_field_x);
        data_collection.AddData(direct_field_y);
        data_collection.AddData(direct_field_z);
        data_collection.AddData(field_error_x);
        data_collection.AddData(field_error_y);
        data_collection.AddData(field_error_z);
        data_collection.AddData(l2_field_error);
        data_collection.AddData(logl2_field_error);

        //timing data
        data_collection.AddData(fmm_time_per_potential_call);
        data_collection.AddData(fmm_time_per_field_call);
        data_collection.AddData(direct_time_per_potential_call);
        data_collection.AddData(direct_time_per_field_call);


        auto* data = new KSAObjectOutputNode<KFMNamedScalarDataCollection>("data_collection");
        data->AttachObjectToNode(&data_collection);

        bool result;
        KEMFileInterface::GetInstance()->SaveKSAFile(data,
                                                     std::string("./test_fast_multipole_field_solver.ksa"),
                                                     result);

#ifdef KEMFIELD_USE_ROOT

        std::cout << "starting root plotting" << std::endl;

        //ROOT stuff for plots
        TApplication* App = new TApplication("TestFastMultipoleFieldSolver", &argc, argv);
        TStyle* myStyle = new TStyle("Plain", "Plain");
        myStyle->SetCanvasBorderMode(0);
        myStyle->SetPadBorderMode(0);
        myStyle->SetPadColor(0);
        myStyle->SetCanvasColor(0);
        myStyle->SetTitleColor(1);
        myStyle->SetPalette(1, 0);        // nice color scale for z-axis
        myStyle->SetCanvasBorderMode(0);  // gets rid of the stupid raised edge around the canvas
        myStyle->SetTitleFillColor(0);    //turns the default dove-grey background to white
        myStyle->SetCanvasColor(0);
        myStyle->SetPadColor(0);
        myStyle->SetTitleFillColor(0);
        myStyle->SetStatColor(0);  //this one may not work
        const int NRGBs = 5;
        const int NCont = 48;
        double stops[NRGBs] = {0.00, 0.34, 0.61, 0.84, 1.00};
        double red[NRGBs] = {0.00, 0.00, 0.87, 1.00, 0.51};
        double green[NRGBs] = {0.00, 0.81, 1.00, 0.20, 0.00};
        double blue[NRGBs] = {0.51, 1.00, 0.12, 0.00, 0.00};
        TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
        myStyle->SetNumberContours(NCont);
        myStyle->cd();


        //plotting objects
        std::vector<TCanvas*> canvas;
        std::vector<TGraph*> graph;
        std::vector<TGraph2D*> graph2d;

        if (mode == 0 || mode == 1 || mode == 2) {
            //collect the rest of the (non-point) scalar data
            unsigned int n_data_sets = data_collection.GetNDataSets();
            for (unsigned int i = 0; i < n_data_sets; i++) {
                std::string name = data_collection.GetDataSetWithIndex(i)->GetName();

                if ((name != (std::string("x_coordinate"))) && (name != (std::string("y_coordinate"))) &&
                    (name != (std::string("z_coordinate")))) {
                    if ((name != (std::string("fmm_time_per_potential_call"))) &&
                        (name != (std::string("fmm_time_per_field_call"))) &&
                        (name != (std::string("direct_time_per_potential_call"))) &&
                        (name != (std::string("direct_time_per_field_call")))) {

                        std::cout << "making graph for " << name << std::endl;

                        TCanvas* c = new TCanvas(name.c_str(), name.c_str(), 50, 50, 950, 850);
                        c->cd();
                        c->SetFillColor(0);
                        c->SetRightMargin(0.2);

                        TGraph* g = new TGraph();
                        std::string title = name + " vs. ";

                        if (mode == 0) {
                            title += "x position";
                        };
                        if (mode == 1) {
                            title += "y position";
                        };
                        if (mode == 2) {
                            title += "z position";
                        };

                        g->SetTitle(title.c_str());
                        graph.push_back(g);

                        for (unsigned int j = 0; j < n_points; j++) {
                            if (mode == 0) {
                                g->SetPoint(j,
                                            x_coord.GetValue(j),
                                            data_collection.GetDataSetWithIndex(i)->GetValue(j));
                            };
                            if (mode == 1) {
                                g->SetPoint(j,
                                            y_coord.GetValue(j),
                                            data_collection.GetDataSetWithIndex(i)->GetValue(j));
                            };
                            if (mode == 2) {
                                g->SetPoint(j,
                                            z_coord.GetValue(j),
                                            data_collection.GetDataSetWithIndex(i)->GetValue(j));
                            };
                        }

                        g->Draw("ALP");
                        c->Update();

                        canvas.push_back(c);
                    }
                }
            }
        }

        if (mode == 3 || mode == 4 || mode == 5) {
            //collect the rest of the (non-point) scalar data
            unsigned int n_data_sets = data_collection.GetNDataSets();
            for (unsigned int i = 0; i < n_data_sets; i++) {
                std::string name = data_collection.GetDataSetWithIndex(i)->GetName();

                if ((name != (std::string("x_coordinate"))) && (name != (std::string("y_coordinate"))) &&
                    (name != (std::string("z_coordinate")))) {
                    if ((name != (std::string("fmm_time_per_potential_call"))) &&
                        (name != (std::string("fmm_time_per_field_call"))) &&
                        (name != (std::string("direct_time_per_potential_call"))) &&
                        (name != (std::string("direct_time_per_field_call")))) {

                        std::cout << "making graph for " << name << std::endl;

                        TCanvas* c = new TCanvas(name.c_str(), name.c_str(), 50, 50, 950, 850);
                        c->SetFillColor(0);
                        c->SetRightMargin(0.2);

                        TGraph2D* g = new TGraph2D(n_points);
                        std::string title = name + " in ";

                        if (mode == 3) {
                            title += "y-z plane";
                        };
                        if (mode == 4) {
                            title += "x-z plane";
                        };
                        if (mode == 5) {
                            title += "x-y plane";
                        };

                        g->SetTitle(title.c_str());
                        graph2d.push_back(g);

                        for (unsigned int j = 0; j < n_points; j++) {
                            if (mode == 3) {
                                g->SetPoint(j,
                                            y_coord.GetValue(j),
                                            z_coord.GetValue(j),
                                            data_collection.GetDataSetWithIndex(i)->GetValue(j));
                            };
                            if (mode == 4) {
                                g->SetPoint(j,
                                            x_coord.GetValue(j),
                                            z_coord.GetValue(j),
                                            data_collection.GetDataSetWithIndex(i)->GetValue(j));
                            };
                            if (mode == 5) {
                                g->SetPoint(j,
                                            x_coord.GetValue(j),
                                            y_coord.GetValue(j),
                                            data_collection.GetDataSetWithIndex(i)->GetValue(j));
                            };
                        }

                        g->SetMarkerStyle(24);
                        g->Draw("PCOLZ");
                        c->Update();

                        canvas.push_back(c);
                    }
                }
            }
        }

        App->Run();

#endif

    }  //end mpi single process

#if KEMFIELD_USE_MPI
    KMPIInterface::GetInstance()->Finalize();
#endif

    return 0;
}
