#include <getopt.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

#include "KGBox.hh"
#include "KGRectangle.hh"
#include "KGMesher.hh"

#include "KGBEM.hh"
#include "KGBEMConverter.hh"

#include "KTypelist.hh"
#include "KSurfaceTypes.hh"
#include "KSurface.hh"
#include "KSurfaceContainer.hh"

#include "KDataDisplay.hh"

#include "KElectrostaticBoundaryIntegrator.hh"
#include "KBoundaryIntegralMatrix.hh"
#include "KBoundaryIntegralVector.hh"
#include "KBoundaryIntegralSolutionVector.hh"

#include "KGaussianElimination.hh"
#include "KRobinHood.hh"
#include "KBiconjugateGradientStabilized.hh"

#include "KIterativeStateWriter.hh"
#include "KIterationTracker.hh"

#include "KSADataStreamer.hh"
#include "KBinaryDataStreamer.hh"

#include "KEMConstants.hh"


#include "KFMElectrostaticSurfaceConverter.hh"
#include "KFMElectrostaticElement.hh"
#include "KFMElectrostaticElementContainer.hh"

#include "KFMElectrostaticTree.hh"
#include "KFMElectrostaticTreeManager.hh"
#include "KFMElectrostaticLocalCoefficientFieldCalculator.hh"
#include "KFMElectrostaticLocalCoefficientCalculatorNumeric.hh"

#include "KFMElectrostaticFastMultipoleFieldSolver.hh"
#include "KFMElectrostaticDirectSubsetFieldSolver.hh"

#include "KFMElectrostaticLocalCoefficientFieldCalculator.hh"
#include "KFMElectrostaticDirectSubsetFieldSolver.hh"

#include "KFMBoundaryIntegralMatrix.hh"
#include "KFMElectrostaticBoundaryIntegrator.hh"

#ifdef KEMFIELD_USE_OPENCL
#include "KFMElectrostaticTreeManager_OpenCL.hh"
#endif


#include <iostream>
#include <iomanip>



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
#include "TLine.h"
#include "TEllipse.h"
#endif

using namespace KGeoBag;
using namespace KEMField;

int degree;
int zmask;
int divisions;

KFMElectrostaticFastMultipoleFieldSolver* solver;
KFMElectrostaticElementContainer<3,1>* elementContainer;
KFMElectrostaticSurfaceConverter* converter;
KFMElectrostaticTree* e_tree;

#ifdef KEMFIELD_USE_OPENCL
KFMElectrostaticTreeManager<KFMElectrostaticTreeManager_OpenCL>* treeManager;
#else
KFMElectrostaticTreeManager<KFMElectrostaticTreeManager_SingleThread>* treeManager;
#endif

KFMElectrostaticTreeNavigator* navigator;
KFMElectrostaticLocalCoefficientFieldCalculator* fieldCalc;
KFMElectrostaticNode* root_node;
KFMElectrostaticIdentitySetCollector* idCollector;
KFMElectrostaticLocalCoefficientFieldCalculator* calc;
KFMElectrostaticLocalCoefficientCalculatorNumeric* fLocalCoeffCalcNumeric;
KFMElectrostaticDirectSubsetFieldSolver* subset_solver;
KIntegratingFieldSolver<KElectrostaticBoundaryIntegrator>* integratingEFieldSolver;




double FMMPotential(double *x, double*)
{
    double point[3];
    point[0] = x[0];
    point[1] = x[1];

    return solver->Potential(point);
}

double PotentialDifference(double *x, double*)
{
    double point[3];
    point[0] = x[0];
    point[1] = x[1];
    point[2] = 0;


    double fmm_pot = solver->Potential(point);

    //return fmm_pot;

    double direct_pot = integratingEFieldSolver->Potential(KPosition(point));

//    return direct_pot;
    return std::max( std::log10( std::fabs(fmm_pot - direct_pot) ), -20. );
}



double FMMPotential2(double* x, double*)
{

    KFMElectrostaticLocalCoefficientSet fNodeMoments;
    fNodeMoments.SetDegree(degree);
    fNodeMoments.Clear();
    KFMElectrostaticLocalCoefficientSet fTempExpansion;
    fTempExpansion.SetDegree(degree);
    fTempExpansion.Clear();

    double point[3];
    point[0] = x[0];
    point[1] = x[1];
    point[2] = 0;

    KFMPoint<3> temp_point(point);
    navigator->SetPoint(&temp_point);
    navigator->ApplyAction(e_tree->GetRootNode());
    std::vector< KFMNode<KFMElectrostaticNodeObjects>* >* nodes;
    nodes = navigator->GetNodeList();

    KFMCube<3>* subregion = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCube<3> >::GetNodeObject(nodes->at(0));
    KFMPoint<3> center = subregion->GetCenter();



    //collect the id's of all the elements that must NOT have their local coefficients evaluated
    std::vector< KFMElectrostaticNode* > nodeNeighborList;
    idCollector->Clear();
    for(unsigned int i=0; i<nodes->size(); i++)
    {
        KFMCubicSpaceNodeNeighborFinder<3, KFMElectrostaticNodeObjects>::GetAllNeighbors( (*nodes)[i], zmask, &nodeNeighborList);

        for(unsigned int j=0; j<nodeNeighborList.size(); j++)
        {
            idCollector->ApplyAction( nodeNeighborList[j] );
        }
    }

    unsigned int N_elements = elementContainer->GetNElements();
    for(unsigned int i=0; i<N_elements; i++)
    {
        if( !(idCollector->GetIDSet()->IsPresent(i) ) )
        {
            fTempExpansion.Clear();
            fLocalCoeffCalcNumeric->ConstructExpansion(center, elementContainer->GetPointCloud(i), &fTempExpansion);

            //need to multiply the temp moments by the charge density here
            double cd = ( *(elementContainer->GetBasisData(i)) )[0];
            fTempExpansion.MultiplyByScalar(cd);
            fNodeMoments += fTempExpansion;
        }
    }

    fieldCalc->SetExpansionOrigin(center);
    fieldCalc->SetLocalCoefficients(&fNodeMoments);
    fieldCalc->SetPoint(point);

    return fieldCalc->Potential();
}



int main(int argc, char** argv)
{

    (void) argc;
    (void) argv;

    // Construct the shape
    KGBox* box = new KGBox();


    int meshCount = 2;

    box->SetX0(-.5);
    box->SetX1(.5);
    box->SetXMeshCount(meshCount+1);
    box->SetXMeshPower(3);

    box->SetY0(-.5);
    box->SetY1(.5);
    box->SetYMeshCount(meshCount+2);
    box->SetYMeshPower(3);

    box->SetZ0(-.5);
    box->SetZ1(.5);
    box->SetZMeshCount(meshCount+3);
    box->SetZMeshPower(3);

    KGSurface* cube = new KGSurface(box);
    cube->SetName("box");
    cube->MakeExtension<KGMesh>();
    cube->MakeExtension<KGElectrostaticDirichlet>();
    cube->AsExtension<KGElectrostaticDirichlet>()->SetBoundaryValue(1.);

    // Mesh the elements
    KGMesher* mesher = new KGMesher();
    cube->AcceptNode(mesher);

    KSurfaceContainer surfaceContainer;
    KGBEMMeshConverter geometryConverter(surfaceContainer);
    cube->AcceptNode(&geometryConverter);

    KElectrostaticBoundaryIntegrator integrator;
    KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A(surfaceContainer,integrator);
    KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x(surfaceContainer,integrator);
    KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b(surfaceContainer,integrator);

    KGaussianElimination<KElectrostaticBoundaryIntegrator::ValueType> gaussianElimination;
    gaussianElimination.Solve(A,x,b);


//    KBiconjugateGradientStabilized<KElectrostaticBoundaryIntegrator::ValueType> conjugateGradient;
//    conjugateGradient.Solve(A,x,b);


     //now we have a surface container with a bunch of electrode discretizations
    //we just want to convert these into point clouds, and then bounding balls
    //extract the information we want
    elementContainer = new KFMElectrostaticElementContainer<3,1>();
    converter = new KFMElectrostaticSurfaceConverter();
    converter->SetSurfaceContainer(&surfaceContainer);
    converter->SetElectrostaticElementContainer(elementContainer);
    converter->Extract();

    //build the tree
    e_tree = new KFMElectrostaticTree();

    //create the tree manager
    #ifdef KEMFIELD_USE_OPENCL
    treeManager = new KFMElectrostaticTreeManager<KFMElectrostaticTreeManager_OpenCL>();
    #else
    treeManager = new KFMElectrostaticTreeManager<KFMElectrostaticTreeManager_SingleThread>();
    #endif

    treeManager->SetTree(e_tree);

    //set some parameters
    degree = 4;
    zmask = 1;
    divisions = 3;
    treeManager->SetDegree(degree);
    treeManager->SetDivisions(divisions);
    treeManager->SetZeroMaskSize(zmask);
    treeManager->SetMaximumTreeDepth(3);
    treeManager->SetRegionSizeFactor(4.1);

    //set the element container
    treeManager->SetElectrostaticElementContainer(elementContainer);

    //intialize
    treeManager->Initialize();

    //build stuff
    treeManager->ConstructRootNode();
    std::cout<<"done constructing root node"<<std::endl;
    treeManager->PerformSpatialSubdivision();
    std::cout<<"done subdivision"<<std::endl;
    treeManager->AssociateElementsAndNodes();
    std::cout<<"done element node association"<<std::endl;
    treeManager->RemoveMultipoleMoments();
    std::cout<<"done removing any pre-existing multipole moments"<<std::endl;
    treeManager->ComputeMultipoleMoments();
    std::cout<<"done computing multipole moments"<<std::endl;
    treeManager->PerformAdjacencySubdivision();
    std::cout<<"done adjacency progenation"<<std::endl;
    treeManager->InitializeLocalCoefficients();
    std::cout<<"done initializing local coefficients"<<std::endl;
    treeManager->ComputeLocalCoefficients();
    std::cout<<"done computing local coefficients"<<std::endl;

    navigator = new KFMElectrostaticTreeNavigator();
    KFMPoint<3> test_point;
    test_point[0] = 0.0001;
    test_point[1] = 0.0001;
    test_point[2] = 0.0001;

    navigator->SetDivisions(divisions);
    navigator->SetPoint(&test_point);
    navigator->ApplyAction(e_tree->GetRootNode());

    std::vector< KFMNode<KFMElectrostaticNodeObjects>* >* nodes;
    nodes = navigator->GetNodeList();


    KFMCube<3>* subregion = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCube<3> >::GetNodeObject(nodes->at(0));
    KFMPoint<3> center = subregion->GetCenter();

    KFMPoint<3> point = center;
    double length = subregion->GetLength();
    point[0] += length/3.0;
    point[1] -= length/4.0;
    point[2] += length/5.0;
    integratingEFieldSolver = new KIntegratingFieldSolver<KElectrostaticBoundaryIntegrator>(surfaceContainer,integrator);

    subset_solver = new KFMElectrostaticDirectSubsetFieldSolver();
    subset_solver->SetIntegratingFieldSolver(integratingEFieldSolver);
   // subset_solver->SetElementContainer(elementContainer);
    subset_solver->Initialize();

    solver = new KFMElectrostaticFastMultipoleFieldSolver();
    solver->SetTree(e_tree);
    solver->SetDegree(degree);
    solver->SetZeroMaskSize(zmask);
    solver->SetDirectFieldCalculator(subset_solver);
    solver->DoNotUseCaching();
    //solver->SetPoint(point);

    root_node = e_tree->GetRootNode();
    KFMCube<3>* root_cube = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCube<3> >::GetNodeObject(root_node);
    double root_length = root_cube->GetLength();
    KFMPoint<3> root_center = root_cube->GetCenter();
    KFMPoint<3> root_low_corner = root_cube->GetCorner(0);
    KFMPoint<3> root_high_corner = root_cube->GetCorner(7);

    std::cout<<"root side length = "<<root_length<<std::endl;
    std::cout<<"root center = ("<<root_center[0]<<", "<<root_center[1]<<", "<<root_center[2]<<std::endl;
    std::cout<<"root low corner = ("<<root_low_corner[0]<<", "<<root_low_corner[1]<<", "<<root_low_corner[2]<<std::endl;
    std::cout<<"root high corner = ("<<root_high_corner[0]<<", "<<root_high_corner[1]<<", "<<root_high_corner[2]<<std::endl;


    #ifdef KEMFIELD_USE_ROOT

    double len = root_length - 0.01;


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


    double xlow = -len/2. + root_center[0];
    double xhigh = len/2. + root_center[0];
    double ylow = -len/2. + root_center[1];
    double yhigh = len/2. + root_center[1];

    //function we want to plot
    TF2* p_fmm = new TF2("fmm_potential", PotentialDifference, xlow, xhigh, ylow, yhigh, 0);
    //set number of points to evaluate at in each direction
    p_fmm->SetNpx(200);
    p_fmm->SetNpy(200);

    TCanvas* canvas = new TCanvas("potential","potential", 50, 50, 950, 850);
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
   // p_fmm->GetZaxis()->SetRangeUser(min, max); // ... set the range ...

    //set the range
    p_fmm->SetMinimum(-21.);
    p_fmm->SetMaximum(0.);


    canvas->Update();
    App->Run();

    #endif


//    delete e_tree;
//    delete treeManager;
//    delete calc;

    return 0;
}
