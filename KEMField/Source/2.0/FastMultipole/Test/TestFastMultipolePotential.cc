#include <getopt.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

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

#include "KDataDisplay.hh"

#include "KElectrostaticBoundaryIntegrator.hh"
#include "KBoundaryIntegralMatrix.hh"
#include "KBoundaryIntegralVector.hh"
#include "KBoundaryIntegralSolutionVector.hh"

#include "KGaussianElimination.hh"
#include "KRobinHood.hh"
#include "KBiconjugateGradientStabilized.hh"
#include "KBiconjugateGradientStabilizedJacobiPreconditioned_SingleThread.hh"

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
#include "KFMElectrostaticParameters.hh"

#include "KFMElectrostaticFastMultipoleFieldSolver.hh"
z

#include "KFMBoundaryIntegralMatrix.hh"
#include "KFMElectrostaticBoundaryIntegrator.hh"

#include "KFMElectrostaticTreeManager.hh"

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


KIntegratingFieldSolver<KElectrostaticBoundaryIntegrator>* direct_solver;

#ifdef KEMFIELD_USE_OPENCL
KFMElectrostaticBoundaryIntegrator<KFMElectrostaticTreeManager_OpenCL>* fast_multipole_boundary_integrator;
#else
KFMElectrostaticBoundaryIntegrator<KFMElectrostaticTreeManager_SingleThread>* fast_multipole_boundary_integrator;
#endif

KFMElectrostaticFastMultipoleFieldSolver* fast_multipole_solver;
KFMElectrostaticDirectSubsetFieldSolver* subset_solver;


double FMMPotential(double *x, double*)
{
    double point[3];
    point[0] = x[0];
    point[1] = x[1];
    point[2] = 0;
    //return direct_solver->Potential(KPosition(point));
    double p = fast_multipole_solver->Potential(point);
    //std::cout<<"p = "<<p<<std::endl;
    return p;
}

double PotentialDifference(double *x, double*)
{
    double point[3];
    point[0] = x[0];
    point[1] = x[1];
    point[2] = 0;

    double fmm_pot = fast_multipole_solver->Potential(point);
    double direct_pot =  direct_solver->Potential(KPosition(point));

//    std::cout<<"fmm_pot = "<<fmm_pot<<std::endl;
//    std::cout<<"direct_pot = "<<direct_pot<<std::endl;

    return std::max( std::log( std::fabs(fmm_pot - direct_pot) ), -20. );
}

int main(int argc, char** argv)
{
    (void) argc;
    (void) argv;

    int use_box = 1;

    KSurfaceContainer surfaceContainer;

    if(use_box == 1)
    {

        // Construct the shape
        KGBox* box = new KGBox();
        int meshCount = 10;

        box->SetX0(-.5);
        box->SetX1(.5);
        box->SetXMeshCount(meshCount+1);
        box->SetXMeshPower(2);

        box->SetY0(-.5);
        box->SetY1(.5);
        box->SetYMeshCount(meshCount+2);
        box->SetYMeshPower(2);

        box->SetZ0(-.5);
        box->SetZ1(.5);
        box->SetZMeshCount(meshCount+3);
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
    else
    {
        int scale = 30;


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

        KGBEMMeshConverter geometryConverter(surfaceContainer);
        geometryConverter.SetMinimumArea(1.e-12);
        hemisphere1->AcceptNode(&geometryConverter);
        hemisphere2->AcceptNode(&geometryConverter);
    }

    std::cout<<"n elements in surface container = "<<surfaceContainer.size()<<std::endl;




    //solve the geometry with gaussian elimination
    KElectrostaticBoundaryIntegrator integrator;
    KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A(surfaceContainer,integrator);
    KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x(surfaceContainer,integrator);
    KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b(surfaceContainer,integrator);
    KGaussianElimination<KElectrostaticBoundaryIntegrator::ValueType> gaussianElimination;
    gaussianElimination.Solve(A,x,b);

    std::cout<<"done gaussian elimination"<<std::endl;

    //create the direct field solver
    direct_solver = new KIntegratingFieldSolver<KElectrostaticBoundaryIntegrator>(surfaceContainer,integrator);

    //now we are going to make the fast multipole boundary integrator (does not solve geometry, just evaluates)
    KFMElectrostaticParameters params;
    params.divisions = 3;
    params.degree = 4;
    params.zeromask = 1;
    params.maximum_tree_depth = 3;
    params.region_expansion_factor = 4.5;

    #ifdef KEMFIELD_USE_OPENCL
    fast_multipole_boundary_integrator = new KFMElectrostaticBoundaryIntegrator<KFMElectrostaticTreeManager_OpenCL>(surfaceContainer);
    #else
    fast_multipole_boundary_integrator = new KFMElectrostaticBoundaryIntegrator<KFMElectrostaticTreeManager_SingleThread>(surfaceContainer);
    #endif
    fast_multipole_boundary_integrator->Initialize(params);



    subset_solver = new KFMElectrostaticDirectSubsetFieldSolver();
    subset_solver->SetIntegratingFieldSolver(direct_solver);
    subset_solver->Initialize();

    fast_multipole_solver = new KFMElectrostaticFastMultipoleFieldSolver();
    fast_multipole_solver->SetTree(fast_multipole_boundary_integrator->GetTree());
    fast_multipole_solver->SetDegree(params.degree);
    fast_multipole_solver->SetZeroMaskSize(params.zeromask);
    fast_multipole_solver->SetDirectFieldCalculator(subset_solver);
    fast_multipole_solver->DoNotUseCaching();



    std::cout<<"done constructing tree/solver"<<std::endl;



    KFMPoint<3> root_center;
    root_center[0] = 0.0;
    root_center[0] = 0.0;
    root_center[0] = 0.0;

    #ifdef KEMFIELD_USE_ROOT

    double len = 1;

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
    TF2* p_fmm = new TF2("error", PotentialDifference, xlow, xhigh, ylow, yhigh, 0);
    //set number of points to evaluate at in each direction
    p_fmm->SetNpx(50);
    p_fmm->SetNpy(50);

    TCanvas* canvas = new TCanvas("potential_err","potential_err", 50, 50, 950, 850);
    canvas->SetFillColor(0);
    canvas->SetRightMargin(0.2);

    p_fmm->GetXaxis()->SetTitle("X");
    p_fmm->GetXaxis()->CenterTitle();
    p_fmm->GetXaxis()->SetTitleOffset(1.2);
    p_fmm->GetYaxis()->SetTitle("Y");
    p_fmm->GetYaxis()->CenterTitle();
    p_fmm->GetYaxis()->SetTitleOffset(1.25);
    p_fmm->GetZaxis()->SetTitle("Abs Error ");
    p_fmm->GetZaxis()->CenterTitle();
    p_fmm->GetZaxis()->SetTitleOffset(1.6);

    //set the range
    p_fmm->SetMinimum(-21.);
    p_fmm->SetMaximum(0.);


    p_fmm->DrawClone("COLZ");// draw "axes", "contents", "statistics box"


    canvas->Update();
    App->Run();

    #endif

    return 0;
}
