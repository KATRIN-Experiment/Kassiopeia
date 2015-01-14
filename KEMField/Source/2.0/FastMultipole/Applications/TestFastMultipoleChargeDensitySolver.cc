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

#include "KBiconjugateGradientStabilized.hh"
#include "KBiconjugateGradientStabilizedJacobiPreconditioned_SingleThread.hh"
#include "KGeneralizedMinimalResidual.hh"
#include "KGeneralizedMinimalResidual_SingleThread.hh"

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


#ifdef KEMFIELD_USE_OPENCL
#include "KFMElectrostaticTreeManager_OpenCL.hh"
#include "KOpenCLSurfaceContainer.hh"
#include "KOpenCLElectrostaticBoundaryIntegrator.hh"
#include "KOpenCLElectrostaticIntegratingFieldSolver.hh"
#endif

#include <iostream>
#include <iomanip>


#ifdef KEMFIELD_USE_OPENCL
#include "KOpenCLSurfaceContainer.hh"
#include "KOpenCLElectrostaticBoundaryIntegrator.hh"
#include "KOpenCLBoundaryIntegralMatrix.hh"
#include "KOpenCLBoundaryIntegralVector.hh"
#include "KOpenCLBoundaryIntegralSolutionVector.hh"
#include "KRobinHood_OpenCL.hh"
#endif


#ifdef KEMFIELD_USE_VTK
#include "KVTKResidualGraph.hh"
#include "KVTKIterationPlotter.hh"
#endif

using namespace KGeoBag;
using namespace KEMField;

KFMElectrostaticFastMultipoleFieldSolver* fast_solver;

int main(int argc, char** argv)
{

    (void) argc;
    (void) argv;

    int use_box = 0;

    KSurfaceContainer surfaceContainerA;
    KSurfaceContainer surfaceContainerB;

    double accuracy = 1e-8;

    if(use_box == 1)
    {

        // Construct the shape
        KGBox* box = new KGBox();
        int meshCount = 1;

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
        box->SetZMeshCount(50*meshCount);
        box->SetZMeshPower(2);

        KGSurface* cube = new KGSurface(box);
        cube->SetName("box");
        cube->MakeExtension<KGMesh>();
        cube->MakeExtension<KGElectrostaticDirichlet>();
        cube->AsExtension<KGElectrostaticDirichlet>()->SetBoundaryValue(1.);

        // Mesh the elements
        KGMesher* mesher = new KGMesher();
        cube->AcceptNode(mesher);

        KGBEMMeshConverter geometryConverterA(surfaceContainerA);
        cube->AcceptNode(&geometryConverterA);

        KGBEMMeshConverter geometryConverterB(surfaceContainerB);
        cube->AcceptNode(&geometryConverterB);

    }
    else
    {
        int scale = 200;

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

        KGBEMMeshConverter geometryConverterA(surfaceContainerA);
        geometryConverterA.SetMinimumArea(1.e-12);
        hemisphere1->AcceptNode(&geometryConverterA);
        hemisphere2->AcceptNode(&geometryConverterA);


        KGBEMMeshConverter geometryConverterB(surfaceContainerB);
        geometryConverterB.SetMinimumArea(1.e-12);
        hemisphere1->AcceptNode(&geometryConverterB);
        hemisphere2->AcceptNode(&geometryConverterB);

    }

    std::cout<<"n elements in surface container A = "<<surfaceContainerA.size()<<std::endl;
    std::cout<<"n elements in surface container B = "<<surfaceContainerB.size()<<std::endl;


    //solve charge densities of surface container A with robin hood

    #ifdef KEMFIELD_USE_OPENCL
        KOpenCLSurfaceContainer oclSurfaceContainer(surfaceContainerA);
        KOpenCLElectrostaticBoundaryIntegrator direct_integrator(oclSurfaceContainer);
        KBoundaryIntegralMatrix<KOpenCLBoundaryIntegrator<KOpenCLElectrostaticBoundaryIntegrator::Basis> > direct_A(oclSurfaceContainer,direct_integrator);
        KBoundaryIntegralVector<KOpenCLBoundaryIntegrator<KOpenCLElectrostaticBoundaryIntegrator::Basis> > direct_b(oclSurfaceContainer,direct_integrator);
        KBoundaryIntegralSolutionVector<KOpenCLBoundaryIntegrator<KOpenCLElectrostaticBoundaryIntegrator::Basis> > direct_x(oclSurfaceContainer,direct_integrator);
    #else
        KElectrostaticBoundaryIntegrator direct_integrator;
        KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> direct_A(surfaceContainerA, direct_integrator);
        KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> direct_x(surfaceContainerA, direct_integrator);
        KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> direct_b(surfaceContainerA, direct_integrator);
    #endif


    #ifdef KEMFIELD_USE_OPENCL
        KRobinHood<KElectrostaticBoundaryIntegrator::ValueType, KRobinHood_OpenCL> robinHood;
    #else
        KRobinHood<KElectrostaticBoundaryIntegrator::ValueType> robinHood;
    #endif

    robinHood.AddVisitor(new KIterationDisplay<KElectrostaticBoundaryIntegrator::ValueType>());
    robinHood.SetTolerance(accuracy);
    robinHood.SetResidualCheckInterval(100);
    robinHood.Solve(direct_A,direct_x,direct_b);



//    #ifdef KEMFIELD_USE_OPENCL
//        KOpenCLSurfaceContainer oclSurfaceContainerB(surfaceContainerB);
//        KOpenCLElectrostaticBoundaryIntegrator directb_integrator(oclSurfaceContainerB);
//        KBoundaryIntegralMatrix<KOpenCLBoundaryIntegrator<KOpenCLElectrostaticBoundaryIntegrator::Basis> > directb_A(oclSurfaceContainerB,directb_integrator);
//        KBoundaryIntegralVector<KOpenCLBoundaryIntegrator<KOpenCLElectrostaticBoundaryIntegrator::Basis> > directb_b(oclSurfaceContainerB,directb_integrator);
//        KBoundaryIntegralSolutionVector<KOpenCLBoundaryIntegrator<KOpenCLElectrostaticBoundaryIntegrator::Basis> > directb_x(oclSurfaceContainerB,directb_integrator);
//    #else
//        KElectrostaticBoundaryIntegrator directb_integrator;
//        KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> directb_A(surfaceContainerB, directb_integrator);
//        KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> directb_x(surfaceContainerB, directb_integrator);
//        KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> directb_b(surfaceContainerB, directb_integrator);
//    #endif


//    #ifdef KEMFIELD_USE_OPENCL
//        KRobinHood<KElectrostaticBoundaryIntegrator::ValueType, KRobinHood_OpenCL> robinHoodb;
//    #else
//        KRobinHood<KElectrostaticBoundaryIntegrator::ValueType> robinHoodb;
//    #endif

//    robinHoodb.AddVisitor(new KIterationDisplay<KElectrostaticBoundaryIntegrator::ValueType>());
//    robinHoodb.SetTolerance(1e-2);
//    robinHoodb.SetResidualCheckInterval(10);
//    robinHoodb.Solve(directb_A,directb_x,directb_b);



    //now we want to construct the tree
    KFMElectrostaticParameters params;
    params.divisions = 3;
    params.degree = 10;
    params.zeromask = 1;
    params.maximum_tree_depth = 3;
    params.region_expansion_factor = 2.1;
    params.use_region_estimation = true;
    params.use_caching = true;
    params.verbosity = 2;


#ifndef KEMFIELD_USE_OPENCL

    typedef KFMElectrostaticBoundaryIntegrator<KFMElectrostaticTreeManager_SingleThread> KFMSingleThreadEBI;
    KFMSingleThreadEBI* fm_integrator = new KFMSingleThreadEBI(surfaceContainerB);
    fm_integrator->SetUniqueIDString(std::string("cd_test"));
    fm_integrator->Initialize(params);
    KFMBoundaryIntegralMatrix< KFMSingleThreadEBI > fm_A(surfaceContainerB, *fm_integrator);
    KBoundaryIntegralSolutionVector< KFMSingleThreadEBI > fm_x(surfaceContainerB, *fm_integrator);
    KBoundaryIntegralVector< KFMSingleThreadEBI > fm_b(surfaceContainerB, *fm_integrator);

    KGeneralizedMinimalResidual< KFMSingleThreadEBI::ValueType, KGeneralizedMinimalResidual_SingleThread> gmres;
    gmres.SetTolerance(accuracy);
    gmres.SetRestartParameter(30);

    gmres.AddVisitor(new KIterationDisplay<double>());

    gmres.Solve(fm_A,fm_x,fm_b);

    delete fm_integrator;


#else


    typedef KFMElectrostaticBoundaryIntegrator<KFMElectrostaticTreeManager_OpenCL> KFMOpenCLEBI;
    KFMOpenCLEBI* fm_integrator = new KFMOpenCLEBI(surfaceContainerB);
    fm_integrator->SetUniqueIDString(std::string("cd_test"));
    fm_integrator->Initialize(params);
    KFMBoundaryIntegralMatrix< KFMOpenCLEBI > fm_A(surfaceContainerB, *fm_integrator);
    KBoundaryIntegralSolutionVector<KFMOpenCLEBI > fm_x(surfaceContainerB, *fm_integrator);
    KBoundaryIntegralVector< KFMOpenCLEBI> fm_b(surfaceContainerB, *fm_integrator);


    KBiconjugateGradientStabilized< KFMOpenCLEBI::ValueType, KBiconjugateGradientStabilizedJacobiPreconditioned_SingleThread> biCGSTAB;

        biCGSTAB.SetTolerance(accuracy);
        biCGSTAB.AddVisitor(new KIterationDisplay<double>());
        biCGSTAB.Solve(fm_A,fm_x,fm_b);


//    KGeneralizedMinimalResidual< KFMOpenCLEBI::ValueType, KGeneralizedMinimalResidual_SingleThread> gmres;
//    gmres.SetTolerance(accuracy);
//    gmres.SetRestartParameter(30);

//    gmres.AddVisitor(new KIterationDisplay<double>());

//    gmres.Solve(fm_A,fm_x,fm_b);

    delete fm_integrator;

#endif

    //now compute the L2 norm difference between the charge densities
    unsigned int n_elem = surfaceContainerA.size();
    double diff_sum = 0;
    double l1_sum = 0;
    double inf_norm = 0;
    double c_sumA = 0;
    double c_sumB = 0;
    double pot_sum = 0;
    double pot_diff_sum = 0;

    std::cout<<std::setprecision(15);


    KElectrostaticBoundaryIntegrator integratorA;
    KIntegratingFieldSolver<KElectrostaticBoundaryIntegrator>* direct_solverA = new KIntegratingFieldSolver<KElectrostaticBoundaryIntegrator>(surfaceContainerA,integratorA);

    KElectrostaticBoundaryIntegrator integratorB;
    KIntegratingFieldSolver<KElectrostaticBoundaryIntegrator>* direct_solverB = new KIntegratingFieldSolver<KElectrostaticBoundaryIntegrator>(surfaceContainerB,integratorB);


    //create a tree
    KFMElectrostaticTree* e_tree = new KFMElectrostaticTree();

    //set the tree parameters
    e_tree->SetParameters(params);

    #ifndef KEMFIELD_USE_OPENCL
        KFMElectrostaticTreeConstructor<KFMElectrostaticTreeManager_SingleThread> constructor;
        constructor.ConstructTree(surfaceContainerB, *e_tree);
    #else
        KFMElectrostaticTreeConstructor<KFMElectrostaticTreeManager_OpenCL> constructor;
        constructor.ConstructTree(surfaceContainerB, *e_tree);
    #endif


    KFMElectrostaticFastMultipoleFieldSolver* fast_solver = new KFMElectrostaticFastMultipoleFieldSolver(surfaceContainerB, *e_tree);


    for(unsigned int i = 0; i<n_elem; i++)
    {

        KTriangle* triA = dynamic_cast<KTriangle*>(surfaceContainerA[i]);
        double areaA = triA->Area();
        KElectrostaticBasis* basisA = dynamic_cast<KElectrostaticBasis*>(surfaceContainerA[i]);
        double cdA = basisA->GetSolution();

        KTriangle* triB = dynamic_cast<KTriangle*>(surfaceContainerB[i]);
        double areaB = triB->Area();
        KElectrostaticBasis* basisB = dynamic_cast<KElectrostaticBasis*>(surfaceContainerB[i]);
        double cdB = basisB->GetSolution();

        double cA = areaA*cdA;
        double cB = areaB*cdB;

        if(areaA != areaB){std::cout<<"areas not equal! comparison invalid."<<std::endl;}

        std::cout<<"charge A @ "<<i<<" = "<<cA<<std::endl;
        std::cout<<"charge B @ "<<i<<" = "<<cB<<std::endl;
        std::cout<<"charge difference @ "<<i<<" = "<<(cA - cB)<<std::endl;

        double potA = direct_solverA->Potential(triA->Centroid());
        double potB = direct_solverB->Potential(triB->Centroid());
        double potC = fast_solver->Potential(triB->Centroid());

        double pot_diff = potA - potB;

        std::cout<<"potential A @ "<<i<<" = "<<potA<<std::endl;
        std::cout<<"potential B @ "<<i<<" = "<<potB<<std::endl;
        std::cout<<"potential C @ "<<i<<" = "<<potC<<std::endl;
        std::cout<<"potential difference AB @ "<<i<<" = "<<pot_diff<<std::endl;
        std::cout<<"potential difference AC @ "<<i<<" = "<<(potA - potC)<<std::endl;
        std::cout<<"potential difference BC @ "<<i<<" = "<<(potB - potC)<<std::endl;

        pot_diff_sum += pot_diff*pot_diff;

        diff_sum += (cA - cB)*(cA - cB);
        l1_sum += std::fabs(cA-cB);
        if(std::fabs(cA-cB) > inf_norm){inf_norm = std::fabs(cA-cB);};
        c_sumA += std::fabs(cA);
        c_sumB += std::fabs(cB);
    }

    diff_sum = std::sqrt(diff_sum);
    pot_diff_sum = std::sqrt(pot_diff_sum);

    std::cout<<"Absolute L2 norm difference in solutions = "<<diff_sum<<std::endl;
    std::cout<<"Absolute L1 norm difference in solutions = "<<l1_sum<<std::endl;
    std::cout<<"Absolute infinity norm difference in solutions = "<<inf_norm<<std::endl;
    std::cout<<"Total |charge| in system solution computed by robin hood = "<<c_sumA<<std::endl;
    std::cout<<"Total |charge| in system solution computed by fast multipole = "<<c_sumB<<std::endl;
    std::cout<<"Relative L2 norm difference in solutions = "<<diff_sum/c_sumA<<std::endl;
    std::cout<<"Relative L1 norm difference in solutions = "<<l1_sum/c_sumA<<std::endl;
    std::cout<<"Relative infinity norm difference in solutions = "<<inf_norm/c_sumA<<std::endl;
    std::cout<<"Absolute L2 norm difference in potentials (BC) = "<<pot_diff_sum<<std::endl;


    return 0;
}
