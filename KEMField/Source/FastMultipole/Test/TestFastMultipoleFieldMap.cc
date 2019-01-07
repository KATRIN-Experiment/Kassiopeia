#include <getopt.h>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>

#include "KThreeVector_KEMField.hh"


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

#include "KEMFileInterface.hh"
#include "KDataDisplay.hh"

#include "KElectrostaticBoundaryIntegratorFactory.hh"
#include "KBoundaryIntegralMatrix.hh"
#include "KBoundaryIntegralVector.hh"
#include "KBoundaryIntegralSolutionVector.hh"
#include "KElectrostaticIntegratingFieldSolver.hh"

#include "KGaussianElimination.hh"
#include "KRobinHood.hh"
#include "KEMConstants.hh"

#include "KFMElectrostaticTree.hh"
#include "KFMElectrostaticTreeConstructor.hh"
#include "KFMElectrostaticFastMultipoleFieldSolver.hh"

#include "KFMNamedScalarData.hh"
#include "KFMNamedScalarDataCollection.hh"

#ifdef KEMFIELD_USE_VTK
#include "KVTKResidualGraph.hh"
#include "KVTKIterationPlotter.hh"
#endif

#ifdef KEMFIELD_USE_ROOT
#include "TCanvas.h"
#include "TApplication.h"
#include "TStyle.h"
#include "TColor.h"
#include "TGraph.h"
#include "TGraph2D.h"
#endif

#ifdef KEMFIELD_USE_OPENCL
#include "KFMElectrostaticFieldMapper_OpenCL.hh"
#include "KFMElectrostaticFastMultipoleFieldSolver_OpenCL.hh"
#include "KOpenCLSurfaceContainer.hh"
#include "KOpenCLElectrostaticNumericBoundaryIntegrator.hh"
#include "KOpenCLBoundaryIntegralMatrix.hh"
#include "KOpenCLBoundaryIntegralVector.hh"
#include "KOpenCLBoundaryIntegralSolutionVector.hh"
#include "KRobinHood_OpenCL.hh"
#include "KOpenCLElectrostaticIntegratingFieldSolver.hh"
#endif


using namespace KGeoBag;
using namespace KEMField;

int main(int argc, char** argv)
{

  std::string usage =
    "\n"
    "Usage: TestFastMultipoleFieldMap <options>\n"
    "\n"
    "This program computes the solution of a simple dirichlet problem and compares the fast multipole field to the direct sovler. \n"
    "\tAvailable options:\n"
    "\t -h, --help               (shows this message and exits)\n"
    "\t -a, --accuracy           (absolute tolerance on residual norm)\n"
    "\t -s, --scale              (scale of geometry discretization)\n"
    "\t -g, --geometry           (0: Spherical capacitor, \n"
    "\t                           1: Cubic capacitor, \n"
    "\t                           2: Cubic and spherical capacitors \n"
    "\t -t, --bem-solver-type    (0: Gaussian elimination, \n"
    "\t                           1: Robin Hood, \n"
    "\t -p, --degree             (degree of multipole expansion) \n"
    "\t -d, --division           (number of spatial divisions) \n"
    "\t -z, --zeromask           (size of zero mask)\n"
    "\t -m, --max-tree-depth     (maximum depth of multipole tree) \n"
    "\t -n, --n-evaluations      (number of evaluations in field map along each dimension) \n"
    "\t -e, --mode               (0: evaluate points along x axis \n"
    "\t                           1: evaluate points along y axis \n"
    "\t                           2: evaluate points along z axis \n"
    "\t                           3: evaluate points in y-z plane \n"
    "\t                           4: evaluate points in x-z plane \n"
    "\t                           5: evaluate points in x-y plane \n"
    ;

    double tolerance = 1e-4;
    unsigned int scale = 20;
    unsigned int geometry = 0;
    unsigned int type = 1;
    unsigned int n_evaluations = 50;
    unsigned int degree = 4;
    unsigned int divisions = 3;
    unsigned int zeromask = 1;
    unsigned int max_tree_depth = 3;
    unsigned int mode = 5;

    static struct option longOptions[] =
    {
        {"help", no_argument, 0, 'h'},
        {"accuracy", required_argument, 0, 'a'},
        {"scale", required_argument, 0, 's'},
        {"geometry", required_argument, 0, 'g'},
        {"bem-solver-type", required_argument, 0, 't'},
        {"n-evaluations", required_argument, 0, 'n'},
        {"degree", required_argument, 0, 'p'},
        {"division", required_argument, 0, 'd'},
        {"zeromask", required_argument, 0, 'z'},
        {"max-tree-depth", required_argument, 0, 'm'},
        {"mode", required_argument, 0, 'e'}
    };

    static const char *optString = "ha:s:g:t:n:p:d:z:m:e:";

    while(1)
    {
        char optId = getopt_long(argc, argv,optString, longOptions, NULL);
        if(optId == -1) break;
        switch(optId)
        {
            case('h'): // help
            std::cout<<usage<<std::endl;
            return 0;
            case('a'):
            tolerance = atof(optarg);
            break;
            case('s'):
            scale = atoi(optarg);
            break;
            case('g'):
            geometry = atoi(optarg);
            break;
            case('t'):
            type = atoi(optarg);
            break;
            case('n'):
            n_evaluations = atoi(optarg);
            break;
            case('p'):
            degree = atoi(optarg);
            break;
            case('d'):
            divisions = atoi(optarg);
            break;
            case('z'):
            zeromask = atoi(optarg);
            break;
            case('m'):
            max_tree_depth = atoi(optarg);
            break;
            case('e'):
            mode = atoi(optarg);
            break;
            default:
            std::cout<<usage<<std::endl;
            return 1;
        }
    }

    KSurfaceContainer surfaceContainer;

    if(geometry == 0 || geometry == 2)
    {
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

    if(geometry == 1 || geometry == 2)
    {
        // Construct the shape
        KGBox* box = new KGBox();
        int meshCount = scale;

//        box->SetX0(-.25);
//        box->SetX1(.25);
//        box->SetXMeshCount(1.5*meshCount+1);
//        box->SetXMeshPower(2);

//        box->SetY0(-.25);
//        box->SetY1(.25);
//        box->SetYMeshCount(1.5*meshCount+2);
//        box->SetYMeshPower(2);

//        box->SetZ0(-.25);
//        box->SetZ1(.25);
//        box->SetZMeshCount(1.5*meshCount+3);
//        box->SetZMeshPower(2);

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
        cube->AsExtension<KGElectrostaticDirichlet>()->SetBoundaryValue(-1.0);

        // Mesh the elements
        KGMesher* mesher = new KGMesher();
        cube->AcceptNode(mesher);

        KGBEMMeshConverter geometryConverter(surfaceContainer);
        cube->AcceptNode(&geometryConverter);
    }

    std::cout<<"Number of elements in BEM problem = "<<surfaceContainer.size()<<std::endl;

    //define fast multipole parameters
    KFMElectrostaticParameters params;
    params.divisions = divisions;
    params.degree = 0;
    params.zeromask = zeromask;
    params.maximum_tree_depth = max_tree_depth;
    params.region_expansion_factor = 1.1;
    params.use_region_estimation = true;
    params.use_caching = true;
    params.verbosity = 3;

    int fHashMaskedBits = 20;
    double fHashThreshold = 1.e-14;

    // compute hash of the bare geometry
    KMD5HashGenerator tShapeHashGenerator;
    tShapeHashGenerator.MaskedBits( fHashMaskedBits );
    tShapeHashGenerator.Threshold( fHashThreshold );
    tShapeHashGenerator.Omit( Type2Type< KElectrostaticBasis >() );
    tShapeHashGenerator.Omit( Type2Type< KBoundaryType< KElectrostaticBasis, KDirichletBoundary > >() );
    string tShapeHash = tShapeHashGenerator.GenerateHash( surfaceContainer );

    std::cout<<"<shape> hash is <" << tShapeHash << ">" << std::endl;

    // compute hash of the parameter values
    KMD5HashGenerator parameterHashGenerator;
    parameterHashGenerator.MaskedBits( fHashMaskedBits );
    parameterHashGenerator.Threshold( fHashThreshold );
    string parameterHash = parameterHashGenerator.GenerateHash( params );

    std::cout<<"<parameter> hash is <" << parameterHash << ">" << std::endl;

    //construct a unique id by stripping the first 6 characters from the shape and parameter hashes
    std::string unique_id = tShapeHash.substr(0,6) + parameterHash.substr(0,6);

    std::cout<<"<unique_id> is <" << unique_id << ">" << std::endl;

    //now we set the degree of the expansion, (we artificially set it to zero before) since
    //we do not want this to contribute to the hash of of the paramters
    //effecting the sparse matrix element labeling, because it is irrelevant for that purpose
    params.degree = degree;

    //define the linear algebra problem
    #ifdef KEMFIELD_USE_OPENCL
    //create opencl container
    KOpenCLData* ocl_data = KOpenCLInterface::GetInstance()->GetActiveData();
    KOpenCLSurfaceContainer* oclSurfaceContainer = NULL;
    if( ocl_data )
    {
        oclSurfaceContainer = dynamic_cast< KOpenCLSurfaceContainer* >( ocl_data );
    }

    if(oclSurfaceContainer == NULL )
    {
        oclSurfaceContainer = new KOpenCLSurfaceContainer( surfaceContainer );
        KOpenCLInterface::GetInstance()->SetActiveData( oclSurfaceContainer );
    }

    KOpenCLElectrostaticNumericBoundaryIntegrator direct_integrator(*oclSurfaceContainer);
    KBoundaryIntegralMatrix<KOpenCLBoundaryIntegrator<KElectrostaticBasis> > A(*oclSurfaceContainer,direct_integrator);
    KBoundaryIntegralVector<KOpenCLBoundaryIntegrator<KElectrostaticBasis> > b(*oclSurfaceContainer,direct_integrator);
    KBoundaryIntegralSolutionVector<KOpenCLBoundaryIntegrator<KElectrostaticBasis> > x(*oclSurfaceContainer,direct_integrator);
    #else
    //standard
    KElectrostaticBoundaryIntegrator direct_integrator {KEBIFactory::MakeDefault()};
    KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A(surfaceContainer,direct_integrator);
    KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x(surfaceContainer,direct_integrator);
    KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b(surfaceContainer,direct_integrator);
    #endif

    switch(type)
    {
        case 0: //gassian elimination
            {
            KGaussianElimination<KElectrostaticBoundaryIntegrator::ValueType> gaussianElimination;
            gaussianElimination.Solve(A,x,b);
            }
        break;
        case 1: //robin hood
            {
            #ifdef KEMFIELD_USE_OPENCL
            KRobinHood<KElectrostaticBoundaryIntegrator::ValueType, KRobinHood_OpenCL> robinHood;
            robinHood.AddVisitor(new KIterationDisplay<KElectrostaticBoundaryIntegrator::ValueType>());
            robinHood.SetTolerance(tolerance);
            robinHood.SetResidualCheckInterval(10);
            robinHood.AddVisitor(new KIterationDisplay<double>());
            robinHood.Solve(A,x,b);
            #else
            KRobinHood<KElectrostaticBoundaryIntegrator::ValueType> robinHood;
            robinHood.AddVisitor(new KIterationDisplay<KElectrostaticBoundaryIntegrator::ValueType>());
            robinHood.SetTolerance(tolerance);
            robinHood.SetResidualCheckInterval(10);
            robinHood.AddVisitor(new KIterationDisplay<double>());
            robinHood.Solve(A,x,b);
            #endif
            }
        break;
    }

    std::cout<<"done charge density solving."<<std::endl;


    //now build the field solvers

    #ifdef KEMFIELD_USE_OPENCL
    //re-create opencl container
    delete oclSurfaceContainer;
    oclSurfaceContainer = NULL;
    KOpenCLInterface::GetInstance()->SetActiveData( oclSurfaceContainer ); //set to NULL

    //rebuild the ocl container
    ocl_data = KOpenCLInterface::GetInstance()->GetActiveData();
    if( ocl_data )
    {
        oclSurfaceContainer = dynamic_cast< KOpenCLSurfaceContainer* >( ocl_data );
    }

    if(oclSurfaceContainer == NULL )
    {
        oclSurfaceContainer = new KOpenCLSurfaceContainer( surfaceContainer );
        KOpenCLInterface::GetInstance()->SetActiveData( oclSurfaceContainer );
    }

    KIntegratingFieldSolver<KOpenCLElectrostaticNumericBoundaryIntegrator>* direct_solver = new KIntegratingFieldSolver<KOpenCLElectrostaticNumericBoundaryIntegrator>(*oclSurfaceContainer, direct_integrator);
    direct_solver->Initialize();
    #else
    KElectrostaticBoundaryIntegrator direct_integrator_single_thread;
    KIntegratingFieldSolver<KElectrostaticBoundaryIntegrator>* direct_solver = new KIntegratingFieldSolver<KElectrostaticBoundaryIntegrator>(surfaceContainer, direct_integrator_single_thread);
    #endif

    //create a tree
    KFMElectrostaticTree* tree = new KFMElectrostaticTree();
    tree->SetParameters(params);

    #ifdef KEMFIELD_USE_OPENCL
    KFMElectrostaticTreeConstructor<KFMElectrostaticFieldMapper_OpenCL> constructor;
    constructor.ConstructTree(*oclSurfaceContainer, *tree);
    #else
    KFMElectrostaticTreeConstructor<KFMElectrostaticFieldMapper_SingleThread> constructor;
    constructor.ConstructTree(surfaceContainer, *tree);
    #endif

    //now build the fast multipole field solver
    #ifdef KEMFIELD_USE_OPENCL
    KFMElectrostaticFastMultipoleFieldSolver_OpenCL* fast_solver = new KFMElectrostaticFastMultipoleFieldSolver_OpenCL(*oclSurfaceContainer, *tree);
    #else
    KFMElectrostaticFastMultipoleFieldSolver* fast_solver = new KFMElectrostaticFastMultipoleFieldSolver(surfaceContainer, *tree);
    #endif

    std::cout<<"starting field evaluation"<<std::endl;


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

    KFMNamedScalarData x_coord; x_coord.SetName("x_coordinate");
    KFMNamedScalarData y_coord; y_coord.SetName("y_coordinate");
    KFMNamedScalarData z_coord; z_coord.SetName("z_coordinate");
    KFMNamedScalarData fmm_potential; fmm_potential.SetName("fast_multipole_potential");
    KFMNamedScalarData direct_potential; direct_potential.SetName("direct_potential");
    KFMNamedScalarData potential_error; potential_error.SetName("potential_error");
    KFMNamedScalarData log_potential_error; log_potential_error.SetName("log_potential_error");

    KFMNamedScalarData fmm_field_x; fmm_field_x.SetName("fast_multipole_field_x");
    KFMNamedScalarData fmm_field_y; fmm_field_y.SetName("fast_multipole_field_y");
    KFMNamedScalarData fmm_field_z; fmm_field_z.SetName("fast_multipole_field_z");

    KFMNamedScalarData direct_field_x; direct_field_x.SetName("direct_field_x");
    KFMNamedScalarData direct_field_y; direct_field_y.SetName("direct_field_y");
    KFMNamedScalarData direct_field_z; direct_field_z.SetName("direct_field_z");

    KFMNamedScalarData field_error_x; field_error_x.SetName("field_error_x");
    KFMNamedScalarData field_error_y; field_error_y.SetName("field_error_y");
    KFMNamedScalarData field_error_z; field_error_z.SetName("field_error_z");

    KFMNamedScalarData l2_field_error; l2_field_error.SetName("l2_field_error");
    KFMNamedScalarData logl2_field_error; logl2_field_error.SetName("logl2_field_error");

    KFMNamedScalarData fmm_time_per_potential_call; fmm_time_per_potential_call.SetName("fmm_time_per_potential_call");
    KFMNamedScalarData fmm_time_per_field_call; fmm_time_per_field_call.SetName("fmm_time_per_field_call");
    KFMNamedScalarData direct_time_per_potential_call; direct_time_per_potential_call.SetName("direct_time_per_potential_call");
    KFMNamedScalarData direct_time_per_field_call; direct_time_per_field_call.SetName("direct_time_per_field_call");

    //compute the positions of the evaluation points
    KFMCube<3>* world = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCube<3> >::GetNodeObject(tree->GetRootNode());
    double world_length = world->GetLength();
    double length_a = world_length/2.0 - 0.001*world_length;
    double length_b = world_length/2.0 - 0.001*world_length;
    KFMPoint<3> center = world->GetCenter();

    KThreeVector direction_a;
    KThreeVector direction_b;
    KThreeVector direction_c;

    KThreeVector p0(center[0], center[1], center[2]);
    KThreeVector point;

    unsigned int n_points = 0;
    KThreeVector* points;

    switch( mode )
    {
        case 0: //x-axis
            n_points = n_evaluations;
            points = new KThreeVector[n_points];
            direction_a[0] = 1.0; direction_a[1] = 0.0; direction_a[2] = 0.0;
            p0 = p0 - length_a*direction_a;
            for(unsigned int i=0; i<n_evaluations; i++)
            {
                point = p0 + i*(2.0*length_a/n_evaluations)*direction_a;
                x_coord.AddNextValue(point[0]);
                y_coord.AddNextValue(point[1]);
                z_coord.AddNextValue(point[2]);
                points[i] = point;
            }
        break;
        case 1: //y-axis
            n_points = n_evaluations;
            points = new KThreeVector[n_points];
            direction_a[0] = 0.0; direction_a[1] = 1.0; direction_a[2] = 0.0;
            p0 = p0 - length_a*direction_a;
            for(unsigned int i=0; i<n_evaluations; i++)
            {
                point = p0 + i*(2.0*length_a/n_evaluations)*direction_a;
                x_coord.AddNextValue(point[0]);
                y_coord.AddNextValue(point[1]);
                z_coord.AddNextValue(point[2]);
                points[i] = point;
            }
        break;
        case 2: //z-axis
            n_points = n_evaluations;
            points = new KThreeVector[n_points];
            direction_a[0] = 0.0; direction_a[1] = 0.0; direction_a[2] = 1.0;
            p0 = p0 - length_a*direction_a;
            for(unsigned int i=0; i<n_evaluations; i++)
            {
                point = p0 + i*(2.0*length_a/n_evaluations)*direction_a;
                x_coord.AddNextValue(point[0]);
                y_coord.AddNextValue(point[1]);
                z_coord.AddNextValue(point[2]);
                points[i] = point;
            }
        break;
        case 3: //y-z plane
            n_points = n_evaluations*n_evaluations;
            points = new KThreeVector[n_points];
            direction_a[0] = 0.0; direction_a[1] = 1.0; direction_a[2] = 0.0;
            direction_b[0] = 0.0; direction_b[1] = 0.0; direction_b[2] = 1.0;
            p0 = p0 - length_a*direction_a;
            p0 = p0 - length_b*direction_b;
            for(unsigned int i=0; i<n_evaluations; i++)
            {
                for(unsigned int j=0; j<n_evaluations; j++)
                {
                    point = p0 + i*(2.0*length_a/n_evaluations)*direction_a + j*(2.0*length_b/n_evaluations)*direction_b;
                    x_coord.AddNextValue(point[0]);
                    y_coord.AddNextValue(point[1]);
                    z_coord.AddNextValue(point[2]);
                    points[ j + n_evaluations*i ] = point;
                }
            }
        break;
        case 4: //x-z plane
            n_points = n_evaluations*n_evaluations;
            points = new KThreeVector[n_points];
            direction_a[0] = 1.0; direction_a[1] = 0.0; direction_a[2] = 0.0;
            direction_b[0] = 0.0; direction_b[1] = 0.0; direction_b[2] = 1.0;
            p0 = p0 - length_a*direction_a;
            p0 = p0 - length_b*direction_b;
            for(unsigned int i=0; i<n_evaluations; i++)
            {
                for(unsigned int j=0; j<n_evaluations; j++)
                {
                    point = p0 + i*(2.0*length_a/n_evaluations)*direction_a + j*(2.0*length_b/n_evaluations)*direction_b;
                    x_coord.AddNextValue(point[0]);
                    y_coord.AddNextValue(point[1]);
                    z_coord.AddNextValue(point[2]);
                    points[ j + n_evaluations*i ] = point;
                }
            }
        break;
        case 5: //x-y plane
            n_points = n_evaluations*n_evaluations;
            points = new KThreeVector[n_points];
            direction_a[0] = 1.0; direction_a[1] = 0.0; direction_a[2] = 0.0;
            direction_b[0] = 0.0; direction_b[1] = 1.0; direction_b[2] = 0.0;
            p0 = p0 - length_a*direction_a;
            p0 = p0 - length_b*direction_b;
            for(unsigned int i=0; i<n_evaluations; i++)
            {
                for(unsigned int j=0; j<n_evaluations; j++)
                {
                    point = p0 + i*(2.0*length_a/n_evaluations)*direction_a + j*(2.0*length_b/n_evaluations)*direction_b;
                    x_coord.AddNextValue(point[0]);
                    y_coord.AddNextValue(point[1]);
                    z_coord.AddNextValue(point[2]);
                    points[ j + n_evaluations*i ] = point;
                }
            }
        break;
    }


    std::cout<<std::setprecision(7);


    //timer
    clock_t start, end;
    double time;

    start = clock();

    //evaluate multipole potential
    for(unsigned int i=0; i<n_points; i++)
    {
        fmm_potential.AddNextValue( fast_solver->Potential(points[i]) );
    }

    end = clock();
    time = ((double)(end - start))/CLOCKS_PER_SEC; // time in seconds
    time /= (double)(n_points);
    std::cout<<"done fmm potential eval"<<std::endl;
    std::cout<<" time per fmm potential evaluation = "<<time<<std::endl;
    fmm_time_per_potential_call.AddNextValue(time);


    start = clock();

    //evaluate multipole field
    //evaluate multipole potential
    for(unsigned int i=0; i<n_points; i++)
    {
        KThreeVector field = fast_solver->ElectricField(points[i]);
        fmm_field_x.AddNextValue(field[0]);
        fmm_field_y.AddNextValue(field[1]);
        fmm_field_z.AddNextValue(field[2]);
    }

    end = clock();
    time = ((double)(end - start))/CLOCKS_PER_SEC; // time in seconds
    time /= (double)(n_points);
    std::cout<<"done fmm field eval"<<std::endl;
    std::cout<<" time per fmm field evaluation = "<<time<<std::endl;
    fmm_time_per_field_call.AddNextValue(time);


    start = clock();

    //evaluate direct potential
    for(unsigned int i=0; i<n_points; i++)
    {
        direct_potential.AddNextValue( direct_solver->Potential(points[i]) );
    }

    end = clock();
    time = ((double)(end - start))/CLOCKS_PER_SEC; // time in seconds
    time /= (double)(n_points);
    std::cout<<"done direct potential eval"<<std::endl;
    std::cout<<" time per direct potential evaluation = "<<time<<std::endl;
    direct_time_per_potential_call.AddNextValue(time);


    start = clock();
    //evaluate direct field
    for(unsigned int i=0; i<n_points; i++)
    {
        KThreeVector field = direct_solver->ElectricField(points[i]);
        direct_field_x.AddNextValue(field[0]);
        direct_field_y.AddNextValue(field[1]);
        direct_field_z.AddNextValue(field[2]);
    }

    end = clock();
    time = ((double)(end - start))/CLOCKS_PER_SEC; // time in seconds
    time /= (double)(n_points);
    std::cout<<"done direct field eval"<<std::endl;
    std::cout<<" time per direct field evaluation = "<<time<<std::endl;
    direct_time_per_field_call.AddNextValue(time);


    //compute the errors
    for(unsigned int i=0; i<n_points; i++)
    {
        unsigned int index = i;
        double pot_err = std::fabs( fmm_potential.GetValue(index) - direct_potential.GetValue(index) );
        potential_error.AddNextValue( pot_err  );
        log_potential_error.AddNextValue( std::max( std::log10(pot_err) , -20. ) );

        double err_x = fmm_field_x.GetValue(index) - direct_field_x.GetValue(index);
        double err_y = fmm_field_y.GetValue(index) - direct_field_y.GetValue(index);
        double err_z = fmm_field_z.GetValue(index) - direct_field_z.GetValue(index);
        double l2_err = std::sqrt(err_x*err_x + err_y*err_y + err_z*err_z);

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


    KSAObjectOutputNode< KFMNamedScalarDataCollection >* data = new KSAObjectOutputNode< KFMNamedScalarDataCollection >("data_collection");
    data->AttachObjectToNode(&data_collection);

    bool result;
    KEMFileInterface::GetInstance()->SaveKSAFile(data, std::string("./test_fast_multipole_field_solver.ksa"), result, true);

    #ifdef KEMFIELD_USE_ROOT

    std::cout<<"starting root plotting"<<std::endl;

    //ROOT stuff for plots
    TApplication* App = new TApplication("TestFastMultipoleFielsSolver",&argc,argv);
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

    //plotting objects
    std::vector< TCanvas* > canvas;
    std::vector< TGraph* > graph;
    std::vector< TGraph2D* > graph2d;

    if(mode == 0 || mode == 1 || mode == 2)
    {
        //collect the rest of the (non-point) scalar data
        unsigned int n_data_sets = data_collection.GetNDataSets();
        for(unsigned int i=0; i<n_data_sets; i++)
        {
            std::string name = data_collection.GetDataSetWithIndex(i)->GetName();

            if( (name != (std::string("x_coordinate") ) ) &&
                (name != (std::string("y_coordinate") ) ) &&
                (name != (std::string("z_coordinate") ) ) )
            {
                if( (name != (std::string("fmm_time_per_potential_call") ) ) &&
                    (name != (std::string("fmm_time_per_field_call") ) ) &&
                    (name != (std::string("direct_time_per_potential_call") ) ) &&
                    (name != (std::string("direct_time_per_field_call") ) )   )
                {

                    std::cout<<"making graph for "<<name<<std::endl;

                    TCanvas* c = new TCanvas(name.c_str(),name.c_str(), 50, 50, 950, 850);
                    c->SetFillColor(0);
                    c->SetRightMargin(0.2);

                    TGraph* g = new TGraph();
                    std::string title = name + " vs. ";

                    if(mode == 0 ){title += "x position";};
                    if(mode == 1 ){title += "y position";};
                    if(mode == 2 ){title += "z position";};

                    g->SetTitle(title.c_str());
                    graph.push_back(g);

                    for(unsigned int j=0; j<n_points; j++)
                    {
                        if(mode == 0 ){g->SetPoint(j, x_coord.GetValue(j), data_collection.GetDataSetWithIndex(i)->GetValue(j) ); };
                        if(mode == 1 ){g->SetPoint(j, y_coord.GetValue(j), data_collection.GetDataSetWithIndex(i)->GetValue(j) ); };
                        if(mode == 2 ){g->SetPoint(j, z_coord.GetValue(j), data_collection.GetDataSetWithIndex(i)->GetValue(j) ); };
                    }

                    g->Draw("ALP");
                    c->Update();

                    canvas.push_back(c);
                }
            }
        }
    }

    if(mode == 3 || mode == 4 || mode ==5 )
    {
        //collect the rest of the (non-point) scalar data
        unsigned int n_data_sets = data_collection.GetNDataSets();
        for(unsigned int i=0; i<n_data_sets; i++)
        {
            std::string name = data_collection.GetDataSetWithIndex(i)->GetName();

            if( (name != (std::string("x_coordinate") ) ) &&
                (name != (std::string("y_coordinate") ) ) &&
                (name != (std::string("z_coordinate") ) ) )
            {
                if( (name != (std::string("fmm_time_per_potential_call") ) ) &&
                    (name != (std::string("fmm_time_per_field_call") ) ) &&
                    (name != (std::string("direct_time_per_potential_call") ) ) &&
                    (name != (std::string("direct_time_per_field_call") ) )   )
                {

                    std::cout<<"making graph for "<<name<<std::endl;

                    TCanvas* c = new TCanvas(name.c_str(),name.c_str(), 50, 50, 950, 850);
                    c->SetFillColor(0);
                    c->SetRightMargin(0.2);

                    TGraph2D* g = new TGraph2D(n_points);
                    std::string title = name + " in ";

                    if(mode == 3 ){title += "y-z plane";};
                    if(mode == 4 ){title += "x-z plane";};
                    if(mode == 5 ){title += "x-y plane";};

                    g->SetTitle(title.c_str());
                    graph2d.push_back(g);

                    for(unsigned int j=0; j<n_points; j++)
                    {
                        if(mode == 3 ){g->SetPoint(j, y_coord.GetValue(j), z_coord.GetValue(j), data_collection.GetDataSetWithIndex(i)->GetValue(j) ); };
                        if(mode == 4 ){g->SetPoint(j, x_coord.GetValue(j), z_coord.GetValue(j), data_collection.GetDataSetWithIndex(i)->GetValue(j) ); };
                        if(mode == 5 ){g->SetPoint(j, x_coord.GetValue(j), y_coord.GetValue(j), data_collection.GetDataSetWithIndex(i)->GetValue(j) ); };
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

    delete direct_solver;
    delete fast_solver;
    delete tree;

    return 0;
}
