#include "KBoundaryIntegralMatrix.hh"
#include "KBoundaryIntegralSolutionVector.hh"
#include "KBoundaryIntegralVector.hh"
#include "KDataDisplay.hh"
#include "KEMConstants.hh"
#include "KEMFileInterface.hh"
#include "KElectrostaticBoundaryIntegratorFactory.hh"
#include "KElectrostaticIntegratingFieldSolver.hh"
#include "KFMBoundaryIntegralMatrix.hh"
#include "KFMElectrostaticBoundaryIntegrator.hh"
#include "KFMElectrostaticFastMultipoleFieldSolver.hh"
#include "KFMElectrostaticFastMultipoleMultipleTreeFieldSolver.hh"
#include "KFMElectrostaticTree.hh"
#include "KFMElectrostaticTreeConstructor.hh"
#include "KFMNamedScalarData.hh"
#include "KFMNamedScalarDataCollection.hh"
#include "KGBEM.hh"
#include "KGBEMConverter.hh"
#include "KGBox.hh"
#include "KGMesher.hh"
#include "KGRectangle.hh"
#include "KGRotatedObject.hh"
#include "KGaussianElimination.hh"
#include "KGeneralizedMinimalResidual.hh"
#include "KGeneralizedMinimalResidual_SingleThread.hh"
#include "KRobinHood.hh"
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
#include "KFMElectrostaticFastMultipoleFieldSolver_OpenCL.hh"
#include "KFMElectrostaticTreeManager_OpenCL.hh"
#include "KOpenCLElectrostaticIntegratingFieldSolver.hh"
#include "KOpenCLElectrostaticNumericBoundaryIntegrator.hh"
#include "KOpenCLSurfaceContainer.hh"
#endif

#include <iomanip>
#include <iostream>


#ifdef KEMFIELD_USE_VTK
#include "KVTKIterationPlotter.hh"
#include "KVTKResidualGraph.hh"
#endif


#ifdef KEMFIELD_USE_ROOT
#include "TApplication.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TComplex.h"
#include "TEllipse.h"
#include "TF1.h"
#include "TF2.h"
#include "TH2D.h"
#include "TLine.h"
#include "TRandom3.h"
#include "TStyle.h"
#endif

using namespace KGeoBag;
using namespace KEMField;


//#ifdef KEMFIELD_USE_OPENCL
//KIntegratingFieldSolver<KOpenCLElectrostaticNumericBoundaryIntegrator>* direct_solver;
//#else
KIntegratingFieldSolver<KElectrostaticBoundaryIntegrator>* direct_solver;
//#endif


KFMElectrostaticFastMultipoleMultipleTreeFieldSolver* fast_solver;
KFMElectrostaticFastMultipoleFieldSolver* fast_solverA;
KFMElectrostaticFastMultipoleFieldSolver* fast_solverB;

int call_count;

double FMMPotential(double* x, double* /* par */)
{
    double point[3];
    point[0] = x[0];
    point[1] = x[1];
    point[2] = 0;

    return fast_solver->Potential(point);
}

double PotentialDifference(double* x, double* /*par*/)
{
    double point[3];
    point[0] = x[0];
    point[1] = x[1];
    point[2] = 0;

    double fmm_pot = fast_solver->Potential(point);

    double direct_pot = direct_solver->Potential(KPosition(point));

    return std::max(std::log10(std::fabs(fmm_pot - direct_pot)), -20.);
}


double FieldDifference(double* x, double* /*par*/)
{
    double point[3];
    point[0] = x[0];
    point[1] = x[1];
    point[2] = 0;

    KPosition position(point);

    KThreeVector fmm_field = fast_solver->ElectricField(position);
    KThreeVector direct_field = direct_solver->ElectricField(position);
    double del = 0;
    del += (fmm_field[0] - direct_field[0]) * (fmm_field[0] - direct_field[0]);
    del += (fmm_field[1] - direct_field[1]) * (fmm_field[1] - direct_field[1]);
    del += (fmm_field[2] - direct_field[2]) * (fmm_field[2] - direct_field[2]);
    del = std::sqrt(del);
    call_count++;

    return std::max(std::log10(std::fabs(del)), -20.);
}

int main(int argc, char** argv)
{
    call_count = 0;

    (void) argc;
    (void) argv;

    unsigned int NEvaluations = 50;

    int use_box = 0;

    KSurfaceContainer surfaceContainer;

    if (use_box == 1) {

        // Construct the shape
        KGBox* box = new KGBox();
        int meshCount = 1;

        box->SetX0(-.5);
        box->SetX1(.5);
        box->SetXMeshCount(meshCount + 1);
        box->SetXMeshPower(2);

        box->SetY0(-.5);
        box->SetY1(.5);
        box->SetYMeshCount(meshCount + 2);
        box->SetYMeshPower(2);

        box->SetZ0(-.5);
        box->SetZ1(.5);
        box->SetZMeshCount(50 * meshCount);
        box->SetZMeshPower(2);

        KGSurface* cube = new KGSurface(box);
        cube->SetName("box");
        cube->MakeExtension<KGMesh>();
        cube->MakeExtension<KGElectrostaticDirichlet>();
        cube->AsExtension<KGElectrostaticDirichlet>()->SetBoundaryValue(1.);

        // Mesh the elements
        KGMesher* mesher = new KGMesher();
        cube->AcceptNode(mesher);

        KGBEMMeshConverter geometryConverter(surfaceContainer);
        cube->AcceptNode(&geometryConverter);
    }
    else {
        int scale = 200;


        // Construct the shape
        double p1[2], p2[2];
        double radius = 1.;
        KGRotatedObject* hemi1 = new KGRotatedObject(scale, 20);
        p1[0] = -1.;
        p1[1] = 0.;
        p2[0] = 0.;
        p2[1] = 1.;
        hemi1->AddArc(p2, p1, radius, true);

        KGRotatedObject* hemi2 = new KGRotatedObject(scale, 20);
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
    }

    std::cout << "n elements in surface container = " << surfaceContainer.size() << std::endl;


    //    KElectrostaticBoundaryIntegrator integrator {KEBIFactory::MakeDefault()};
    //    KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A(surfaceContainer,integrator);
    //    KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x(surfaceContainer,integrator);
    //    KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b(surfaceContainer,integrator);

    //    KRobinHood<KElectrostaticBoundaryIntegrator::ValueType> robinHood;
    //    robinHood.AddVisitor(new KIterationDisplay<KElectrostaticBoundaryIntegrator::ValueType>());
    //    robinHood.SetTolerance(1e-2);
    //    robinHood.SetResidualCheckInterval(10);
    //    robinHood.Solve(A,x,b);

    //now we want to construct the tree
    KFMElectrostaticParameters params;
    params.divisions = 3;
    params.degree = 6;
    params.zeromask = 1;
    params.maximum_tree_depth = 3;
    params.region_expansion_factor = 2.1;
    params.use_region_estimation = true;
    params.use_caching = true;
    params.verbosity = 2;

#ifndef KEMFIELD_USE_OPENCL

    typedef KFMElectrostaticBoundaryIntegrator<KFMElectrostaticTreeManager_SingleThread> KFMSingleThreadEBI;
    KFMSingleThreadEBI* fm_integrator = new KFMSingleThreadEBI(surfaceContainer);
    fm_integrator->SetUniqueIDString(std::string("test"));
    fm_integrator->Initialize(params);
    KFMBoundaryIntegralMatrix<KFMSingleThreadEBI> A(surfaceContainer, *fm_integrator);
    KBoundaryIntegralSolutionVector<KFMSingleThreadEBI> x(surfaceContainer, *fm_integrator);
    KBoundaryIntegralVector<KFMSingleThreadEBI> b(surfaceContainer, *fm_integrator);

    KGeneralizedMinimalResidual<KFMSingleThreadEBI::ValueType, KGeneralizedMinimalResidual_SingleThread> gmres;
    gmres.SetTolerance(1e-4);
    gmres.SetRestartParameter(30);

    gmres.AddVisitor(new KIterationDisplay<double>());

#ifdef KEMFIELD_USE_VTK
    gmres.AddVisitor(new KVTKIterationPlotter<double>());
#endif

    gmres.Solve(A, x, b);

    delete fm_integrator;


#else


    typedef KFMElectrostaticBoundaryIntegrator<KFMElectrostaticTreeManager_OpenCL> KFMOpenCLEBI;
    KFMOpenCLEBI* fm_integrator = new KFMOpenCLEBI(surfaceContainer);
    fm_integrator->SetUniqueIDString(std::string("test"));
    fm_integrator->Initialize(params);
    KFMBoundaryIntegralMatrix<KFMOpenCLEBI> A(surfaceContainer, *fm_integrator);
    KBoundaryIntegralSolutionVector<KFMOpenCLEBI> x(surfaceContainer, *fm_integrator);
    KBoundaryIntegralVector<KFMOpenCLEBI> b(surfaceContainer, *fm_integrator);

    KGeneralizedMinimalResidual<KFMOpenCLEBI::ValueType, KGeneralizedMinimalResidual_SingleThread> gmres;
    gmres.SetTolerance(1e-4);
    gmres.SetRestartParameter(30);


    gmres.AddVisitor(new KIterationDisplay<double>());

#ifdef KEMFIELD_USE_VTK
    gmres.AddVisitor(new KVTKIterationPlotter<double>());
#endif

    gmres.Solve(A, x, b);

    delete fm_integrator;

#endif


    //    KGaussianElimination<KElectrostaticBoundaryIntegrator::ValueType> gaussianElimination;
    //    gaussianElimination.Solve(A,x,b);

    std::cout << "done charge density solving." << std::endl;

    //    #ifdef KEMFIELD_USE_OPENCL
    KOpenCLSurfaceContainer ocl_container(surfaceContainer);
    //    KOpenCLElectrostaticNumericBoundaryIntegrator integrator(ocl_container);
    //    direct_solver = new KIntegratingFieldSolver<KOpenCLElectrostaticNumericBoundaryIntegrator>(ocl_container,integrator);
    //    #else
    KElectrostaticBoundaryIntegrator integrator{KEBIFactory::MakeDefault()};
    direct_solver = new KIntegratingFieldSolver<KElectrostaticBoundaryIntegrator>(surfaceContainer, integrator);
    //    #endif


    //create a tree
    KFMElectrostaticTree* e_tree = new KFMElectrostaticTree();

    //set the tree parameters
    e_tree->SetParameters(params);

#ifndef KEMFIELD_USE_OPENCL
    KFMElectrostaticTreeConstructor<KFMElectrostaticTreeManager_SingleThread> constructor;
    constructor.ConstructTree(surfaceContainer, *e_tree);
#else
    KFMElectrostaticTreeConstructor<KFMElectrostaticTreeManager_OpenCL> constructor;
    constructor.ConstructTree(surfaceContainer, *e_tree);
#endif

    //create a second tree with different parameters
    KFMElectrostaticParameters params2;
    params2.divisions = 2;
    params2.degree = 6;
    params2.zeromask = 1;
    params2.maximum_tree_depth = 6;
    params2.region_expansion_factor = 2.1;
    params2.use_region_estimation = true;
    params2.use_caching = true;
    params2.verbosity = 2;

    //create a tree
    KFMElectrostaticTree* e_tree2 = new KFMElectrostaticTree();

    //set the tree parameters
    e_tree2->SetParameters(params2);


#ifndef KEMFIELD_USE_OPENCL
    KFMElectrostaticTreeConstructor<KFMElectrostaticTreeManager_SingleThread> constructor2;
    constructor2.ConstructTree(surfaceContainer, *e_tree2);
#else
    KFMElectrostaticTreeConstructor<KFMElectrostaticTreeManager_OpenCL> constructor2;
    constructor2.ConstructTree(surfaceContainer, *e_tree2);
#endif

    //    //now build the fast multipole field solver
    fast_solver = new KFMElectrostaticFastMultipoleMultipleTreeFieldSolver(surfaceContainer);
    fast_solver->AddTree(e_tree);
    fast_solver->AddTree(e_tree2);

    fast_solverA = new KFMElectrostaticFastMultipoleFieldSolver(surfaceContainer, *e_tree);
    fast_solverB = new KFMElectrostaticFastMultipoleFieldSolver(surfaceContainer, *e_tree2);

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
    KFMNamedScalarData fmm_potentialA;
    fmm_potentialA.SetName("fast_multipole_potentialA");
    KFMNamedScalarData fmm_potentialB;
    fmm_potentialB.SetName("fast_multipole_potentialB");
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

    //compute the positions of the evaluation points
    double length_a = 3.0;
    double length_b = 3.0;
    KThreeVector direction_a(1.0, 0.0, 0.0);
    KThreeVector direction_b(0.0, 1.0, 0.0);
    KThreeVector p0(-1.5, -1.5, 0.0);
    KThreeVector point;
    for (unsigned int i = 0; i < NEvaluations; i++) {
        for (unsigned int j = 0; j < NEvaluations; j++) {
            point = p0 + i * (length_a / NEvaluations) * direction_a + j * (length_b / NEvaluations) * direction_b;
            x_coord.AddNextValue(point[0]);
            y_coord.AddNextValue(point[1]);
            z_coord.AddNextValue(point[2]);
        }
    }


    //evaluate multipole potential
    for (unsigned int i = 0; i < NEvaluations; i++) {
        for (unsigned int j = 0; j < NEvaluations; j++) {
            KPosition position;
            position = p0 + i * (length_a / NEvaluations) * direction_a + j * (length_b / NEvaluations) * direction_b;
            fmm_potential.AddNextValue(fast_solver->Potential(position));
            fmm_potentialA.AddNextValue(fast_solverA->Potential(position));
            fmm_potentialB.AddNextValue(fast_solverB->Potential(position));
        }
    }

    std::cout << "done fmm potential eval" << std::endl;

    //evaluate multipole field
    for (unsigned int i = 0; i < NEvaluations; i++) {
        for (unsigned int j = 0; j < NEvaluations; j++) {
            KPosition position;
            position = p0 + i * (length_a / NEvaluations) * direction_a + j * (length_b / NEvaluations) * direction_b;
            KThreeVector field = fast_solver->ElectricField(position);
            fmm_field_x.AddNextValue(field[0]);
            fmm_field_y.AddNextValue(field[1]);
            fmm_field_z.AddNextValue(field[2]);
        }
    }

    std::cout << "done fmm field eval" << std::endl;

    //construct an index list of all the surface elements
    std::vector<unsigned int> surface_list;
    unsigned int total_number_of_surfaces = surfaceContainer.size();
    for (unsigned int i = 0; i < total_number_of_surfaces; i++) {
        surface_list.push_back(i);
    }

    //evaluate direct potential
    for (unsigned int i = 0; i < NEvaluations; i++) {
        for (unsigned int j = 0; j < NEvaluations; j++) {
            KPosition position;
            position = p0 + i * (length_a / NEvaluations) * direction_a + j * (length_b / NEvaluations) * direction_b;
            direct_potential.AddNextValue(direct_solver->Potential(position));
        }
    }

    std::cout << "done direct potential eval" << std::endl;


    //evaluate direct field
    for (unsigned int i = 0; i < NEvaluations; i++) {
        for (unsigned int j = 0; j < NEvaluations; j++) {
            KPosition position;
            position = p0 + i * (length_a / NEvaluations) * direction_a + j * (length_b / NEvaluations) * direction_b;
            KThreeVector field = direct_solver->ElectricField(position);
            direct_field_x.AddNextValue(field[0]);
            direct_field_y.AddNextValue(field[1]);
            direct_field_z.AddNextValue(field[2]);
        }
    }

    std::cout << "done fmm field eval" << std::endl;


    //compute the errors
    for (unsigned int i = 0; i < NEvaluations; i++) {
        for (unsigned int j = 0; j < NEvaluations; j++) {
            unsigned int index = j + i * NEvaluations;
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
    }

    std::cout << "done computing errors" << std::endl;


    KFMNamedScalarDataCollection data_collection;
    data_collection.AddData(x_coord);
    data_collection.AddData(y_coord);
    data_collection.AddData(z_coord);
    data_collection.AddData(fmm_potential);
    data_collection.AddData(fmm_potentialA);
    data_collection.AddData(fmm_potentialB);
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


    KSAObjectOutputNode<KFMNamedScalarDataCollection>* data =
        new KSAObjectOutputNode<KFMNamedScalarDataCollection>("data_collection");
    data->AttachObjectToNode(&data_collection);

    bool result;
    KEMFileInterface::GetInstance()->SaveKSAFile(data, std::string("./test.ksa"), result, true);


#ifdef KEMFIELD_USE_ROOT

    std::cout << "starting root stuff" << std::endl;

    //figure out region to compute the field in
    KFMElectrostaticNode* root_node = e_tree->GetRootNode();
    KFMCube<3>* root_cube = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCube<3>>::GetNodeObject(root_node);
    double root_length = root_cube->GetLength();
    KFMPoint<3> root_center = root_cube->GetCenter();
    KFMPoint<3> root_low_corner = root_cube->GetCorner(0);
    KFMPoint<3> root_high_corner = root_cube->GetCorner(7);

    double len = root_length - 0.01;

    //ROOT stuff for plots
    TApplication* App = new TApplication("ERR", &argc, argv);
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


    double xlow = -len / 2. + root_center[0];
    double xhigh = len / 2. + root_center[0];
    double ylow = -len / 2. + root_center[1];
    double yhigh = len / 2. + root_center[1];

    //function we want to plot
    TF2* p_fmm = new TF2("fmm_potential", PotentialDifference, xlow, xhigh, ylow, yhigh, 0);
    //set number of points to evaluate at in each direction
    p_fmm->SetNpx(100);
    p_fmm->SetNpy(100);

    TCanvas* canvas = new TCanvas("potential", "potential", 50, 50, 950, 850);
    canvas->SetFillColor(0);
    canvas->SetRightMargin(0.2);

    p_fmm->Draw("COLZ");
    p_fmm->GetXaxis()->SetTitle("X");
    p_fmm->GetXaxis()->CenterTitle();
    p_fmm->GetXaxis()->SetTitleOffset(1.2);
    p_fmm->GetYaxis()->SetTitle("Y");
    p_fmm->GetYaxis()->CenterTitle();
    p_fmm->GetYaxis()->SetTitleOffset(1.25);
    p_fmm->GetZaxis()->SetTitle("Potential (V)");
    p_fmm->GetZaxis()->CenterTitle();
    p_fmm->GetZaxis()->SetTitleOffset(1.6);

    //set the range
    p_fmm->SetMinimum(-21.);
    p_fmm->SetMaximum(0.);


    canvas->Update();
    App->Run();

#endif

    delete fast_solver;
    delete e_tree;
    delete e_tree2;

    return 0;
}
