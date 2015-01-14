#include <getopt.h>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>


#include "KGBox.hh"
#include "KGRectangle.hh"
#include "KGRotatedObject.hh"
#include "KGMesher.hh"

#include "KGBEM.hh"
#include "KGBEMConverter.hh"

#include "KTypelist.hh"
#include "KSurfaceTypes.hh"
#include "KSurface.hh"
#include "KSurfaceContainer.hh"

#include "KEMThreeVector.hh"
#include "KEMFileInterface.hh"
#include "KDataDisplay.hh"

#include "KElectrostaticBoundaryIntegrator.hh"
#include "KBoundaryIntegralMatrix.hh"
#include "KBoundaryIntegralVector.hh"
#include "KBoundaryIntegralSolutionVector.hh"


#include "KFMBoundaryIntegralMatrix.hh"
#include "KFMElectrostaticBoundaryIntegrator.hh"
#include "KFMElectrostaticFastMultipoleFieldSolver.hh"

#include "KElectrostaticIntegratingFieldSolver.hh"

#include "KGaussianElimination.hh"
#include "KRobinHood.hh"

#include "KEMConstants.hh"

#include "KFMElectrostaticTree.hh"
#include "KFMElectrostaticTreeConstructor.hh"
#include "KFMElectrostaticFastMultipoleFieldSolver.hh"

#include "KFMNamedScalarData.hh"
#include "KFMNamedScalarDataCollection.hh"


#include "KFMSurfaceSubdivider.hh"
#include "KFMSurfaceSubdivisionElectrostaticTreeCondition.hh"
#include "KFMSurfaceMeshSubdivider.hh"


#ifdef KEMFIELD_USE_OPENCL
#include "KFMElectrostaticTreeManager_OpenCL.hh"
#include "KOpenCLSurfaceContainer.hh"
#include "KOpenCLElectrostaticBoundaryIntegrator.hh"
#include "KOpenCLElectrostaticIntegratingFieldSolver.hh"
#include "KOpenCLElectrostaticBoundaryIntegrator.hh"
#include "KOpenCLBoundaryIntegralMatrix.hh"
#include "KOpenCLBoundaryIntegralVector.hh"
#include "KOpenCLBoundaryIntegralSolutionVector.hh"
#include "KRobinHood_OpenCL.hh"
#endif

#include <iostream>
#include <iomanip>


#ifdef KEMFIELD_USE_VTK
#include "KVTKResidualGraph.hh"
#include "KVTKIterationPlotter.hh"
#endif



#ifdef KEMFIELD_USE_ROOT
#include "TRandom3.h"
#include "TF1.h"
#include "TComplex.h"
#include "TRandom3.h"
#include "TCanvas.h"
#include "TH2D.h"
#include "TApplication.h"
#include "TStyle.h"
#include "TColor.h"
#include "TF2.h"
#include "TH1D.h"
#include "TLine.h"
#include "TEllipse.h"
#endif


using namespace KGeoBag;
using namespace KEMField;

KFMElectrostaticFastMultipoleFieldSolver* fast_solver;
int main(int argc, char** argv)
{
    (void) argc;
    (void) argv;

    unsigned int NEvaluations = 100;

    int use_box = 0;

    KSurfaceContainer primaryContainer;
    KSurfaceContainer secondaryContainer;
    KSurfaceContainer tertiaryContainer;

    if(use_box == 1)
    {

        // Construct the shape
        KGBox* box = new KGBox();
        int meshCount = 20;

        box->SetX0(-.5);
        box->SetX1(.5);
        box->SetXMeshCount(meshCount);
        box->SetXMeshPower(2);

        box->SetY0(-.5);
        box->SetY1(.5);
        box->SetYMeshCount(meshCount);
        box->SetYMeshPower(2);

        box->SetZ0(-.5);
        box->SetZ1(.5);
        box->SetZMeshCount(meshCount);
        box->SetZMeshPower(2);

        KGSurface* cube = new KGSurface(box);
        cube->SetName("box");
        cube->MakeExtension<KGMesh>();
        cube->MakeExtension<KGElectrostaticDirichlet>();
        cube->AsExtension<KGElectrostaticDirichlet>()->SetBoundaryValue(1.);

        // Mesh the elements
        KGMesher* mesher = new KGMesher();
        cube->AcceptNode(mesher);

        KGBEMMeshConverter geometryConverter(primaryContainer);
        cube->AcceptNode(&geometryConverter);
    }
    else
    {
        int scale = 100;


        // Construct the shape
        double p1[2],p2[2];
        double radius = 1.;
        KGRotatedObject* hemi1 = new KGRotatedObject(scale,20);
        p1[0] = -1.; p1[1] = 0.;
        p2[0] = 0.; p2[1] = 1.;
        hemi1->AddArc(p2,p1,radius,true);

        KGRotatedObject* hemi2 = new KGRotatedObject(scale,20);
        p2[0] = 1.; p2[1] = 0.;
        p1[0] = 0.; p1[1] = 1.;
        hemi2->AddArc(p1,p2,radius,false);

        // Construct shape placement
        KGRotatedSurface* h1 = new KGRotatedSurface(hemi1);
        KGSurface* hemisphere1 = new KGSurface(h1);
        hemisphere1->SetName( "hemisphere1" );
        hemisphere1->MakeExtension<KGMesh>();
        hemisphere1->MakeExtension<KGElectrostaticDirichlet>();
        hemisphere1->AsExtension<KGElectrostaticDirichlet>()->SetBoundaryValue(1.);

        KGRotatedSurface* h2 = new KGRotatedSurface(hemi2);
        KGSurface* hemisphere2 = new KGSurface(h2);
        hemisphere2->SetName( "hemisphere2" );
        hemisphere2->MakeExtension<KGMesh>();
        hemisphere2->MakeExtension<KGElectrostaticDirichlet>();
        hemisphere2->AsExtension<KGElectrostaticDirichlet>()->SetBoundaryValue(1.);

        // Mesh the elements
        KGMesher* mesher = new KGMesher();
        hemisphere1->AcceptNode(mesher);
        hemisphere2->AcceptNode(mesher);

        KGBEMMeshConverter geometryConverter(primaryContainer);
        geometryConverter.SetMinimumArea(1.e-12);
        hemisphere1->AcceptNode(&geometryConverter);
        hemisphere2->AcceptNode(&geometryConverter);
    }

    std::cout<<"n elements in primary surface container = "<<primaryContainer.size()<<std::endl;

    //now we want to construct the tree
    KFMElectrostaticParameters params;
    params.divisions = 3;
    params.degree = 0;
    params.zeromask = 1;
    params.maximum_tree_depth = 4;
    params.region_expansion_factor = 2.1;
    params.use_region_estimation = true;
    params.use_caching = true;
    params.verbosity = 2;
    std::string unique_id = "test";

    //construct initial tree
    KFMElectrostaticTree* e_tree = new KFMElectrostaticTree();
    //set the tree parameters
    e_tree->SetParameters(params);

    #ifndef KEMFIELD_USE_OPENCL
        KFMElectrostaticTreeConstructor<KFMElectrostaticTreeManager_SingleThread> constructor;
        constructor.ConstructTree(primaryContainer, *e_tree);
    #else
        KFMElectrostaticTreeConstructor<KFMElectrostaticTreeManager_OpenCL> constructor;
        constructor.ConstructTree(primaryContainer, *e_tree);
    #endif

    KFMSurfaceSubdivisionElectrostaticTreeCondition condition;
    condition.SetElectrostaticTree(e_tree);

    KFMSurfaceMeshSubdivider mesh_subdivider;
    mesh_subdivider.SetSurfaceSubdivisionCondition(&condition);
    mesh_subdivider.SubdivideMesh(primaryContainer, secondaryContainer);

    std::cout<<"n elements in secondary surface container = "<<secondaryContainer.size()<<std::endl;


    KFMElectrostaticNode* root = e_tree->GetRootNode();
    KFMCube<3>* root_cube = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCube<3> >::GetNodeObject(root);
    KFMPoint<3> root_center = root_cube->GetCenter();
    double side_length = root_cube->GetLength();

    //now we want to construct the tree
    params.use_region_estimation = false;
    params.world_center_x = root_center[0];
    params.world_center_y = root_center[1];
    params.world_center_z = root_center[2];
    params.world_length = side_length;

    //construct second tree
    KFMElectrostaticTree* e_tree2 = new KFMElectrostaticTree();
    //set the tree parameters
    e_tree2->SetParameters(params);

    #ifndef KEMFIELD_USE_OPENCL
        KFMElectrostaticTreeConstructor<KFMElectrostaticTreeManager_SingleThread> constructor2;
        constructor.ConstructTree(secondaryContainer, *e_tree2);
    #else
        KFMElectrostaticTreeConstructor<KFMElectrostaticTreeManager_OpenCL> constructor2;
        constructor.ConstructTree(secondaryContainer, *e_tree2);
    #endif


////////////////////////////////////////////////////////////////////////////////

    KFMSurfaceSubdivisionElectrostaticTreeCondition condition3;
    condition3.SetElectrostaticTree(e_tree2);

    KFMSurfaceMeshSubdivider mesh_subdivider3;
    mesh_subdivider3.SetSurfaceSubdivisionCondition(&condition3);

    mesh_subdivider3.SubdivideMesh(secondaryContainer, tertiaryContainer);

    std::cout<<"n elements in tertiaryContainer surface container = "<<tertiaryContainer.size()<<std::endl;


    //construct third tree
    KFMElectrostaticTree* e_tree3 = new KFMElectrostaticTree();
    //set the tree parameters
    e_tree3->SetParameters(params);

    #ifndef KEMFIELD_USE_OPENCL
        KFMElectrostaticTreeConstructor<KFMElectrostaticTreeManager_SingleThread> constructor3;
        constructor.ConstructTree(tertiaryContainer, *e_tree3);
    #else
        KFMElectrostaticTreeConstructor<KFMElectrostaticTreeManager_OpenCL> constructor3;
        constructor.ConstructTree(tertiaryContainer, *e_tree3);
    #endif



////////////////////////////////////////////////////////////////////////////////



    //now we have a surface container with a bunch of electrode discretizations
    //we just want to convert these into point clouds, and then bounding balls
    //extract the information we want
    KFMElectrostaticElementContainer<3,1>* elementContainer = new KFMElectrostaticElementContainer<3,1>();
    KFMElectrostaticSurfaceConverter* converter = new KFMElectrostaticSurfaceConverter();
    converter->SetSurfaceContainer(&primaryContainer);
    converter->SetElectrostaticElementContainer(elementContainer);
    converter->Extract();

    //now we have a surface container with a bunch of electrode discretizations
    //we just want to convert these into point clouds, and then bounding balls
    //extract the information we want
    KFMElectrostaticElementContainer<3,1>* elementContainer2 = new KFMElectrostaticElementContainer<3,1>();
    KFMElectrostaticSurfaceConverter* converter2 = new KFMElectrostaticSurfaceConverter();
    converter->SetSurfaceContainer(&secondaryContainer);
    converter->SetElectrostaticElementContainer(elementContainer2);
    converter->Extract();

    //now we have a surface container with a bunch of electrode discretizations
    //we just want to convert these into point clouds, and then bounding balls
    //extract the information we want
    KFMElectrostaticElementContainer<3,1>* elementContainer3 = new KFMElectrostaticElementContainer<3,1>();
    KFMElectrostaticSurfaceConverter* converter3 = new KFMElectrostaticSurfaceConverter();
    converter->SetSurfaceContainer(&tertiaryContainer);
    converter->SetElectrostaticElementContainer(elementContainer3);
    converter->Extract();


    #ifdef KEMFIELD_USE_ROOT
    //ROOT stuff for plots
    TApplication* App = new TApplication("ERR",&argc,argv);
    TStyle* myStyle = new TStyle("Plain", "Plain");
    myStyle->SetCanvasBorderMode(0);
    myStyle->SetPadBorderMode(0);
    myStyle->SetPadColor(0);
    myStyle->SetCanvasColor(0);
    myStyle->SetTitleColor(1);
    myStyle->SetPalette(1,0);   // nice color scale for z-axis
    myStyle->SetCanvasBorderMode(0); // gets rid of the stupid raised edge around the canvas
    myStyle->SetTitleFillColor(0); //turns the default dove-grey background to white
    myStyle->SetCanvasColor(0);
    myStyle->SetPadColor(0);
    myStyle->SetTitleFillColor(0);
    myStyle->SetStatColor(0); //this one may not work
    const int NRGBs = 5;
    const int NCont = 48;
    double stops[NRGBs] = { 0.00, 0.34, 0.61, 0.84, 1.00 };
    double red[NRGBs]   = { 0.00, 0.00, 0.87, 1.00, 0.51 };
    double green[NRGBs] = { 0.00, 0.81, 1.00, 0.20, 0.00 };
    double blue[NRGBs]  = { 0.51, 1.00, 0.12, 0.00, 0.00 };
    TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
    myStyle->SetNumberContours(NCont);
    myStyle->cd();


    TCanvas* canvas = new TCanvas("canvas","canvas", 50, 50, 950, 850);
    TCanvas* canvas2 = new TCanvas("canvas2","canvas2", 50, 50, 950, 850);
    TCanvas* canvas3 = new TCanvas("canvas3","canvas3", 50, 50, 950, 850);
    canvas->cd();
    canvas->SetFillColor(0);
    canvas->SetRightMargin(0.2);

    //loop over all bounding balls in geometry and histogram the sizes
    TH1* h  = new TH1D("h1", "h1 title", 10000, 1e-6, 1.0);
    unsigned int n = elementContainer->GetNElements();
    std::cout<<"primary container has :"<<n<<std::endl;
    double radius;
    for(unsigned int i=0; i<n; i++)
    {
        radius = elementContainer->GetBoundingBall(i)->GetRadius();
        h->Fill(radius);
    }
    h->Draw();
    canvas->SetLogx();

    canvas2->cd();

    TH1* h2  = new TH1D("h2", "h2 title", 10000, 1e-6, 1.0);
    n = elementContainer2->GetNElements();
    std::cout<<"secondary container has :"<<n<<std::endl;
    for(unsigned int i=0; i<n; i++)
    {
        radius = elementContainer2->GetBoundingBall(i)->GetRadius();
        h2->Fill(radius);
    }
    h2->Draw();
    canvas2->SetLogx();


    canvas3->cd();

    TH1* h3  = new TH1D("h3", "h3 title", 10000, 1e-6, 1.0);
    n = elementContainer2->GetNElements();
    std::cout<<"tertiary container has :"<<n<<std::endl;
    for(unsigned int i=0; i<n; i++)
    {
        radius = elementContainer3->GetBoundingBall(i)->GetRadius();
        h3->Fill(radius);
    }
    h3->Draw();
    canvas3->SetLogx();


    canvas->Update();
    canvas2->Update();
    canvas3->Update();
    App->Run();

    #endif




























//#ifndef KEMFIELD_USE_OPENCL

//    typedef KFMElectrostaticBoundaryIntegrator<KFMElectrostaticTreeManager_SingleThread> KFMSingleThreadEBI;
//    KFMSingleThreadEBI* fm_integrator = new KFMSingleThreadEBI(primaryContainer);
//    fm_integrator->SetUniqueIDString(unique_id);
//    fm_integrator->Initialize(params);
//    KFMBoundaryIntegralMatrix< KFMSingleThreadEBI > A(primaryContainer, *fm_integrator);
//    KBoundaryIntegralSolutionVector< KFMSingleThreadEBI > x(primaryContainer, *fm_integrator);
//    KBoundaryIntegralVector< KFMSingleThreadEBI > b(primaryContainer, *fm_integrator);

//    KGeneralizedMinimalResidual< KFMSingleThreadEBI::ValueType, KGeneralizedMinimalResidual_SingleThread> gmres;
//    gmres.SetTolerance(1e-4);
//    gmres.SetRestartParameter(30);

//    gmres.AddVisitor(new KIterationDisplay<double>());

//#ifdef KEMFIELD_USE_VTK
//    gmres.AddVisitor(new KVTKIterationPlotter<double>());
//#endif

//    gmres.Solve(A,x,b);

//    delete fm_integrator;


//#else


//    typedef KFMElectrostaticBoundaryIntegrator<KFMElectrostaticTreeManager_OpenCL> KFMOpenCLEBI;
//    KFMOpenCLEBI* fm_integrator = new KFMOpenCLEBI(primaryContainer);
//    fm_integrator->SetUniqueIDString(unique_id);
//    fm_integrator->Initialize(params);
//    KFMBoundaryIntegralMatrix< KFMOpenCLEBI > A(primaryContainer, *fm_integrator);
//    KBoundaryIntegralSolutionVector<KFMOpenCLEBI > x(primaryContainer, *fm_integrator);
//    KBoundaryIntegralVector< KFMOpenCLEBI> b(primaryContainer, *fm_integrator);

//    KGeneralizedMinimalResidual< KFMOpenCLEBI::ValueType, KGeneralizedMinimalResidual_SingleThread> gmres;
//    gmres.SetTolerance(1e-4);
//    gmres.SetRestartParameter(30);


//    gmres.AddVisitor(new KIterationDisplay<double>());

//#ifdef KEMFIELD_USE_VTK
//    gmres.AddVisitor(new KVTKIterationPlotter<double>());
//#endif

//    gmres.Solve(A,x,b);

//    delete fm_integrator;

//#endif





////    KGaussianElimination<KElectrostaticBoundaryIntegrator::ValueType> gaussianElimination;
////    gaussianElimination.Solve(A,x,b);

//    std::cout<<"done charge density solving."<<std::endl;

////    #ifdef KEMFIELD_USE_OPENCL
////    KOpenCLprimaryContainer ocl_container(primaryContainer);
////    KOpenCLElectrostaticBoundaryIntegrator integrator(ocl_container);
////    direct_solver = new KIntegratingFieldSolver<KOpenCLElectrostaticBoundaryIntegrator>(ocl_container,integrator);
////    #else
//    KElectrostaticBoundaryIntegrator integrator;
//    direct_solver = new KIntegratingFieldSolver<KElectrostaticBoundaryIntegrator>(primaryContainer,integrator);
////    #endif

//    //create a tree
//    KFMElectrostaticTree* e_tree = new KFMElectrostaticTree();

//    //set the tree parameters
//    e_tree->SetParameters(params);

//    #ifndef KEMFIELD_USE_OPENCL
//        KFMElectrostaticTreeConstructor<KFMElectrostaticTreeManager_SingleThread> constructor;
//        constructor.ConstructTree(primaryContainer, *e_tree);
//    #else
//        KFMElectrostaticTreeConstructor<KFMElectrostaticTreeManager_OpenCL> constructor;
//        constructor.ConstructTree(primaryContainer, *e_tree);
//    #endif


//    //save the tree data
//    KFMElectrostaticTreeData output_data;
//    constructor.SaveTree(*e_tree, output_data);

//    //create a root node
//    std::string treeID = "test";
//    std::string treeFileName = "fm_tree.zksa";
//    KSAObjectOutputNode< KFMElectrostaticTreeData >* tree_data_output_node = new KSAObjectOutputNode< KFMElectrostaticTreeData >(treeID);
//    tree_data_output_node->AttachObjectToNode(&output_data);

//    bool result = false;
//    bool forceOverwrite = true;
//    KEMFileInterface::GetInstance()->SaveKSAFile(tree_data_output_node, treeFileName, result, forceOverwrite);

//    if(result)
//    {
//        std::cout<<"saved electrostatic fast multipole tree to "<<  KEMFileInterface::GetInstance()->ActiveDirectory() + "/" + treeFileName <<std::endl;
//    }
//    else
//    {
//        std::cout<<"failed to save electrostatic fast multipole tree to "<<  KEMFileInterface::GetInstance()->ActiveDirectory() + "/" + treeFileName <<std::endl;
//    }

//    //now we are going to read back the tree data
//    result = false;
//    KSAObjectInputNode< KFMElectrostaticTreeData >* tree_data_input_node = new KSAObjectInputNode< KFMElectrostaticTreeData >(treeID);
//    KEMFileInterface::GetInstance()->ReadKSAFile(tree_data_input_node, treeFileName, result);

//    KFMElectrostaticTree* e_tree2 = new KFMElectrostaticTree();

//    if(result)
//    {
//        std::cout<< "electrostatic fast multipole tree found." <<std::endl;
//        //sucessful read of file, now get data object
//        KFMElectrostaticTreeData* input_data = tree_data_input_node->GetObject();

//        //construct tree from data
//        constructor.ConstructTree(*input_data, *e_tree2);
//    }
//    else
//    {
//        std::cout<<" tree data not found in file: "<<KEMFileInterface::GetInstance()->ActiveDirectory() + "/" + treeFileName<<std::endl;
//    }

//    //now build the fast multipole field solver
//    fast_solver = new KFMElectrostaticFastMultipoleFieldSolver(primaryContainer, *e_tree2);
////    fast_solver = new KFMElectrostaticFastMultipoleFieldSolver(primaryContainer, *e_tree);





//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

//    KFMNamedScalarData x_coord; x_coord.SetName("x_coordinate");
//    KFMNamedScalarData y_coord; y_coord.SetName("y_coordinate");
//    KFMNamedScalarData z_coord; z_coord.SetName("z_coordinate");
//    KFMNamedScalarData fmm_potential; fmm_potential.SetName("fast_multipole_potential");
//    KFMNamedScalarData direct_potential; direct_potential.SetName("direct_potential");
//    KFMNamedScalarData potential_error; potential_error.SetName("potential_error");
//    KFMNamedScalarData log_potential_error; log_potential_error.SetName("log_potential_error");

//    KFMNamedScalarData fmm_field_x; fmm_field_x.SetName("fast_multipole_field_x");
//    KFMNamedScalarData fmm_field_y; fmm_field_y.SetName("fast_multipole_field_y");
//    KFMNamedScalarData fmm_field_z; fmm_field_z.SetName("fast_multipole_field_z");

//    KFMNamedScalarData direct_field_x; direct_field_x.SetName("direct_field_x");
//    KFMNamedScalarData direct_field_y; direct_field_y.SetName("direct_field_y");
//    KFMNamedScalarData direct_field_z; direct_field_z.SetName("direct_field_z");

//    KFMNamedScalarData field_error_x; field_error_x.SetName("field_error_x");
//    KFMNamedScalarData field_error_y; field_error_y.SetName("field_error_y");
//    KFMNamedScalarData field_error_z; field_error_z.SetName("field_error_z");

//    KFMNamedScalarData l2_field_error; l2_field_error.SetName("l2_field_error");
//    KFMNamedScalarData logl2_field_error; logl2_field_error.SetName("logl2_field_error");

//    //compute the positions of the evaluation points
//    double length_a = 3.0;
//    double length_b = 3.0;
//    KEMThreeVector direction_a(1.0, 0.0, 0.0);
//    KEMThreeVector direction_b(0.0, 1.0, 0.0);
//    KEMThreeVector p0(-1.5, -1.5, 0.0);
//    KEMThreeVector point;
//    for(unsigned int i=0; i<NEvaluations; i++)
//    {
//        for(unsigned int j=0; j<NEvaluations; j++)
//        {
//            point = p0 + i*(length_a/NEvaluations)*direction_a + j*(length_b/NEvaluations)*direction_b;
//            x_coord.AddNextValue(point[0]);
//            y_coord.AddNextValue(point[1]);
//            z_coord.AddNextValue(point[2]);
//        }
//    }


//    //evaluate multipole potential
//    for(unsigned int i=0; i<NEvaluations; i++)
//    {
//        for(unsigned int j=0; j<NEvaluations; j++)
//        {
//            KPosition position;
//            position = p0 + i*(length_a/NEvaluations)*direction_a + j*(length_b/NEvaluations)*direction_b;
//            fmm_potential.AddNextValue( fast_solver->Potential(position) );
//        }
//    }

//    std::cout<<"done fmm potential eval"<<std::endl;

//    //evaluate multipole field
//    for(unsigned int i=0; i<NEvaluations; i++)
//    {
//        for(unsigned int j=0; j<NEvaluations; j++)
//        {
//            KPosition position;
//            position = p0 + i*(length_a/NEvaluations)*direction_a + j*(length_b/NEvaluations)*direction_b;
//            KEMThreeVector field = fast_solver->ElectricField(position);
//            fmm_field_x.AddNextValue(field[0]);
//            fmm_field_y.AddNextValue(field[1]);
//            fmm_field_z.AddNextValue(field[2]);
//        }
//    }

//    std::cout<<"done fmm field eval"<<std::endl;

//    //construct an index list of all the surface elements
//    std::vector<unsigned int> surface_list;
//    unsigned int total_number_of_surfaces = primaryContainer.size();
//    for(unsigned int i=0; i<total_number_of_surfaces; i++)
//    {
//        surface_list.push_back(i);
//    }

//    int co = 0;
//    //evaluate direct potential
//    for(unsigned int i=0; i<NEvaluations; i++)
//    {
//        for(unsigned int j=0; j<NEvaluations; j++)
//        {
//            KPosition position;
//            position = p0 + i*(length_a/NEvaluations)*direction_a + j*(length_b/NEvaluations)*direction_b;
//            direct_potential.AddNextValue( direct_solver->Potential(position) );
////            direct_potential.AddNextValue( direct_solver->Potential(&surface_list, position) );
////            std::cout<<"direct eval # "<<co<<" = "<<direct_solver->Potential(position)<<std::endl;
////            std::cout<<"direct eval # "<<co<<" = "<<direct_solver->Potential(&surface_list, position)<<std::endl;
//            co++;
//        }
//    }

//    std::cout<<"done direct potential eval"<<std::endl;


//    //evaluate direct field
//    for(unsigned int i=0; i<NEvaluations; i++)
//    {
//        for(unsigned int j=0; j<NEvaluations; j++)
//        {
//            KPosition position;
//            position = p0 + i*(length_a/NEvaluations)*direction_a + j*(length_b/NEvaluations)*direction_b;
//            KEMThreeVector field = direct_solver->ElectricField(position);
////            KEMThreeVector field = direct_solver->ElectricField(&surface_list, position);
//            direct_field_x.AddNextValue(field[0]);
//            direct_field_y.AddNextValue(field[1]);
//            direct_field_z.AddNextValue(field[2]);
//        }
//    }

//    std::cout<<"done fmm field eval"<<std::endl;


//    //compute the errors
//    for(unsigned int i=0; i<NEvaluations; i++)
//    {
//        for(unsigned int j=0; j<NEvaluations; j++)
//        {
//            unsigned int index = j + i*NEvaluations;
//            double pot_err = std::fabs( fmm_potential.GetValue(index) - direct_potential.GetValue(index) );
//            potential_error.AddNextValue( pot_err  );
//            log_potential_error.AddNextValue( std::max( std::log10(pot_err) , -20. ) );

//            double err_x = fmm_field_x.GetValue(index) - direct_field_x.GetValue(index);
//            double err_y = fmm_field_y.GetValue(index) - direct_field_y.GetValue(index);
//            double err_z = fmm_field_z.GetValue(index) - direct_field_z.GetValue(index);
//            double l2_err = std::sqrt(err_x*err_x + err_y*err_y + err_z*err_z);

//            field_error_x.AddNextValue(err_x);
//            field_error_y.AddNextValue(err_y);
//            field_error_z.AddNextValue(err_z);
//            l2_field_error.AddNextValue(l2_err);
//            logl2_field_error.AddNextValue(std::log10(l2_err));
//        }
//    }

//    std::cout<<"done computing errors"<<std::endl;


//    KFMNamedScalarDataCollection data_collection;
//    data_collection.AddData(x_coord);
//    data_collection.AddData(y_coord);
//    data_collection.AddData(z_coord);
//    data_collection.AddData(fmm_potential);
//    data_collection.AddData(direct_potential);
//    data_collection.AddData(potential_error);
//    data_collection.AddData(log_potential_error);
//    data_collection.AddData(fmm_field_x);
//    data_collection.AddData(fmm_field_y);
//    data_collection.AddData(fmm_field_z);
//    data_collection.AddData(direct_field_x);
//    data_collection.AddData(direct_field_y);
//    data_collection.AddData(direct_field_z);
//    data_collection.AddData(field_error_x);
//    data_collection.AddData(field_error_y);
//    data_collection.AddData(field_error_z);
//    data_collection.AddData(l2_field_error);
//    data_collection.AddData(logl2_field_error);


//    KSAObjectOutputNode< KFMNamedScalarDataCollection >* data = new KSAObjectOutputNode< KFMNamedScalarDataCollection >("data_collection");
//    data->AttachObjectToNode(&data_collection);

//    KEMFileInterface::GetInstance()->SaveKSAFile(data, std::string("./test.ksa"), result, true);

//    delete fast_solver;
    delete e_tree;
    delete e_tree2;

    return 0;
}
