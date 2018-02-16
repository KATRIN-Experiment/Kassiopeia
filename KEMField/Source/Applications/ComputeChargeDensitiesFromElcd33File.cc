// This program computes charge densities from a geometry defined in a
// Elcd33 text file containing rectangles and wires.
// Author: Daniel Hilk
// Date: April 28th, 2016

#include <getopt.h>


#include "KTypelist.hh"
#include "KSurfaceTypes.hh"
#include "KSurface.hh"
#include "KSurfaceContainer.hh"

#include "KDataDisplay.hh"

#include "KSADataStreamer.hh"
#include "KBinaryDataStreamer.hh"
#include "KSerializer.hh"

#include "KElectrostaticBoundaryIntegratorFactory.hh"
#include "KBoundaryIntegralMatrix.hh"
#include "KBoundaryIntegralVector.hh"
#include "KBoundaryIntegralSolutionVector.hh"

#include "KGaussianElimination.hh"
#include "KRobinHood.hh"

#include "KIterativeStateWriter.hh"
#include "KIterationTracker.hh"

#include "KSADataStreamer.hh"
#include "KBinaryDataStreamer.hh"
#include "KEMFileInterface.hh"

#include "KEMConstants.hh"

#ifdef KEMFIELD_USE_MPI
#include "KRobinHood_MPI.hh"
//#define MPI_SINGLE_PROCESS if (KMPIInterface::GetInstance()->GetProcess()==0)
#else
#define MPI_SINGLE_PROCESS
#endif

#ifdef KEMFIELD_USE_OPENCL
#include "KOpenCLSurfaceContainer.hh"
#include "KOpenCLElectrostaticBoundaryIntegratorFactory.hh"
#include "KOpenCLBoundaryIntegralMatrix.hh"
#include "KOpenCLBoundaryIntegralVector.hh"
#include "KOpenCLBoundaryIntegralSolutionVector.hh"
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
#include "KEMVTKViewer.hh"
#include "KVTKIterationPlotter.hh"
#endif

using namespace KEMField;

// typedefs for elements
typedef KSurface<KElectrostaticBasis,KDirichletBoundary,KRectangle> KEMRectangle;
typedef KSurface<KElectrostaticBasis,KDirichletBoundary,KLineSegment> KEMWire;


void AddRectangle( double sideA, double sideB, KPosition p0, KPosition n1, KPosition n2, double pot, KSurfaceContainer &cont )
{
    KEMRectangle* rectangle = new KEMRectangle();

    rectangle->SetA( sideA );
    rectangle->SetB( sideB );
    rectangle->SetP0( p0 );
    rectangle->SetN1( n1 );
    rectangle->SetN2( n2 );

    rectangle->SetBoundaryValue( pot );

    cont.push_back( rectangle );

    return;
}


void AddWire( KPosition pA, KPosition pB, double diameter, double pot, KSurfaceContainer &cont )
{
    KEMWire* wire = new KEMWire();

    wire->SetP0( pA );
    wire->SetP1( pB );
    wire->SetDiameter( diameter);

    wire->SetBoundaryValue( pot );

    cont.push_back( wire );

    return;
}

void ReadElectrodeFile( std::string inputFileName, KSurfaceContainer &container )
{
    std::ifstream inputFileStream( inputFileName.c_str() );
    unsigned int L( 0 );
    unsigned int subelindex( 0 ), groupindex( 0 ), ntype( 0 ), nrot( 0 );
    double par[11], U( 0. );

    KPosition VecP0( 0., 0., 0. );
    KPosition VecN1( 0., 0., 0. );
    KPosition VecN2( 0., 0., 0. );

    KPosition VecPA( 0., 0., 0. );
    KPosition VecPB( 0., 0., 0. );

    if (!inputFileStream )
    {
        KEMField::cout << "Error: File " << inputFileName << " cannot be read." <<  KEMField::endl;
    }
    else
    {
        inputFileStream >> L;

        for( unsigned int i=0; i<L; i++ )
        {
            inputFileStream >> subelindex >> groupindex >> ntype >> nrot;

            for( unsigned int j=0; j<11; j++ ) {
                inputFileStream >> par[j];
            }

            inputFileStream >> U;

            // rectangle
            if( ntype == 1 ) {
                VecP0.SetComponents( par[0], par[1], par[2] );
                VecN1.SetComponents( par[3], par[4], par[5] );
                VecN2.SetComponents( par[6], par[7], par[8] );
                AddRectangle( par[9], par[10], VecP0, VecN1, VecN2, U, container );
            }

            // wire
            if( ntype == 2 ) {
                VecPA.SetComponents( par[0], par[1], par[2] );
                VecPB.SetComponents( par[3], par[4], par[5] );
                AddWire( VecPA, VecPB, par[6], U, container );
            }

        }
    }

    inputFileStream.close();

    return;
}

int main(int argc, char* argv[])
{
#ifdef KEMFIELD_USE_PETSC
  KPETScInterface::GetInstance()->Initialize(&argc,&argv);
#elif KEMFIELD_USE_MPI
  KMPIInterface::GetInstance()->Initialize(&argc,&argv);
#endif

  std::string usage =
    "\n"
    "Usage: MainSpectrometerChargeDensity <options>\n"
    "\n"
    "This program computes the charge density profile of KATRIN's main spectrometer.\n"
    "\n"
    "\tAvailable options:\n"
    "\t -h, --help               (shows this message and exits)\n"
    "\t -v, --verbose            (bool; sets the verbosity)\n"
    "\t -s, --scale              (discretization scale)\n"
    "\t -a, --accuracy           (accuracy of charge density computation)\n"
    "\t -i, --increment          (increment of accuracy check)\n"
    "\t -j, --save-increment     (increment of state save)\n"
    "\t -k, --file            (name of the output file)\n"
#ifdef KEMFIELD_USE_VTK
    "\t -e, --with-plot          (dynamic plot of residual norm)\n"
#endif
    "\t -m, --method             (gauss"
#ifdef KEMFIELD_USE_PETSC
    ", robinhood or PETSc)\n";
#else
  " or robinhood)\n";
#endif

  bool verbose = 1;

  double accuracy = 1.e-8;
  (void) accuracy;
  unsigned int increment = 1000;
  unsigned int saveIncrement = 50000;
  bool usePlot = false;
  int method = 1;

  std::string file = "ChDenFromTxtInput";

  static struct option longOptions[] = {
    {"help", no_argument, 0, 'h'},
    {"verbose", required_argument, 0, 'v'},
    {"accuracy", required_argument, 0, 'a'},
    {"increment", required_argument, 0, 'i'},    
    {"save-increment", required_argument, 0, 'j'},    
    {"file", required_argument, 0, 'k'},
#ifdef KEMFIELD_USE_VTK
    {"with-plot", no_argument, 0, 'e'},
#endif
    {"method", required_argument, 0, 'm'},
  };

#ifdef KEMFIELD_USE_VTK
  static const char *optString = "hv:a:i:j:k:em:";
#else
  static const char *optString = "hv:a:i:j:k:m:";
#endif

  while(1) {
    char optId = getopt_long(argc, argv,optString, longOptions, NULL);
    if(optId == -1) break;
    switch(optId) {
    case('h'): // help
      MPI_SINGLE_PROCESS
	std::cout<<usage<<std::endl;
#ifdef KEMFIELD_USE_MPI
      KMPIInterface::GetInstance()->Finalize();
#endif
      return 0;
    case('v'): // verbose
      verbose = atoi(optarg);
      break;
    case('a'):
      accuracy = atof(optarg);
      break;
    case('i'):
      increment = atoi(optarg);
      break;
    case('j'):
      saveIncrement = atoi(optarg);
      break;
    case('k'):
      file = optarg;
      break;
#ifdef KEMFIELD_USE_VTK
    case('e'):
      usePlot = true;
      break;
#endif
    case('m'):
      method = atoi(optarg);
      break;
    default: // unrecognized option
      MPI_SINGLE_PROCESS
	std::cout<<usage<<std::endl;
#ifdef KEMFIELD_USE_MPI
      KMPIInterface::GetInstance()->Finalize();
#endif
      return 1;
    }
  }

#if defined(KEMFIELD_USE_MPI) && defined(KEMFIELD_USE_OPENCL)
  KOpenCLInterface::GetInstance()->SetGPU(KMPIInterface::GetInstance()->GetProcess());
#endif

  KEMField::cout.Verbose(false);

  MPI_SINGLE_PROCESS
    KEMField::cout.Verbose(verbose);

  KSurfaceContainer surfaceContainer;

  // read elcd33 input file
  ReadElectrodeFile( file, surfaceContainer );

  MPI_SINGLE_PROCESS
  {
    KEMField::cout << "" << KEMField::endl;
    KEMField::cout << "Computing the charge densities of file " << file << KEMField::endl;
    KEMField::cout<<surfaceContainer.size()<<" elements to accuracy " << accuracy << "." << KEMField::endl;
  }

  if (method == 0)
  {
    KElectrostaticBoundaryIntegrator integrator{KEBIFactory::MakeDefault()};
    KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A(surfaceContainer,integrator);
    KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x(surfaceContainer,integrator);
    KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b(surfaceContainer,integrator);

    KGaussianElimination<KElectrostaticBoundaryIntegrator::ValueType> gaussianElimination;
    gaussianElimination.Solve(A,x,b);
  }
  else if (method == 1)
  {
#ifdef KEMFIELD_USE_OPENCL
    KOpenCLSurfaceContainer oclSurfaceContainer(surfaceContainer);
    KOpenCLElectrostaticBoundaryIntegrator integrator{KoclEBIFactory::MakeDefault(oclSurfaceContainer)};
    KBoundaryIntegralMatrix<KOpenCLBoundaryIntegrator<KElectrostaticBasis> > A(oclSurfaceContainer,integrator);
    KBoundaryIntegralVector<KOpenCLBoundaryIntegrator<KElectrostaticBasis> > b(oclSurfaceContainer,integrator);
    KBoundaryIntegralSolutionVector<KOpenCLBoundaryIntegrator<KElectrostaticBasis> > x(oclSurfaceContainer,integrator);
#else
    KElectrostaticBoundaryIntegrator integrator{KEBIFactory::MakeDefault()};
    KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A(surfaceContainer,integrator);
    KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x(surfaceContainer,integrator);
    KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b(surfaceContainer,integrator);
#endif

#if defined(KEMFIELD_USE_MPI) && defined(KEMFIELD_USE_OPENCL)
    KRobinHood<KElectrostaticBoundaryIntegrator::ValueType,
      KRobinHood_MPI_OpenCL> robinHood;
#ifndef KEMFIELD_USE_DOUBLE_PRECISION
    robinHood.SetTolerance((accuracy > 1.e-5 ? accuracy : 1.e-5));
#else
    robinHood.SetTolerance(accuracy);
#endif
#elif defined(KEMFIELD_USE_MPI)
    KRobinHood<KElectrostaticBoundaryIntegrator::ValueType,
      KRobinHood_MPI> robinHood;
#elif defined(KEMFIELD_USE_OPENCL)
    KRobinHood<KElectrostaticBoundaryIntegrator::ValueType,
      KRobinHood_OpenCL> robinHood;
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
      KIterationTracker<KElectrostaticBoundaryIntegrator::ValueType>* tracker = new KIterationTracker<KElectrostaticBoundaryIntegrator::ValueType>();
      tracker->Interval(1);
      tracker->WriteInterval(100);
      tracker->MaxIterationStamps(1.e6);
      robinHood.AddVisitor(tracker);
    }

    KIterativeStateWriter<KElectrostaticBoundaryIntegrator::ValueType>* stateWriter = new KIterativeStateWriter<KElectrostaticBoundaryIntegrator::ValueType>(surfaceContainer);
    stateWriter->Interval(saveIncrement);
    stateWriter->SaveNameRoot(file);
    robinHood.AddVisitor(stateWriter);

#ifdef KEMFIELD_USE_VTK
    MPI_SINGLE_PROCESS
      if (usePlot)
	robinHood.AddVisitor(new KVTKIterationPlotter<KElectrostaticBoundaryIntegrator::ValueType>(5));
#else
    (void)usePlot;
#endif

    robinHood.SetResidualCheckInterval(increment);
    robinHood.Solve(A,x,b);
  }
#ifdef KEMFIELD_USE_PETSC
  else if (method == 2)
  {
    KElectrostaticBoundaryIntegrator integrator{KEBIFactory::MakeDefault()};
    KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A(surfaceContainer,integrator);
    KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x(surfaceContainer,integrator);
    KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b(surfaceContainer,integrator);

    KPETScSolver<KElectrostaticBoundaryIntegrator::ValueType> petscSolver;
    petscSolver.Solve(A,x,b);
  }
#endif

  MPI_SINGLE_PROCESS
  {
    KMetadataStreamer mDS;
    mDS.open(file + std::string(".smd"),"overwrite");
    mDS << surfaceContainer;
    mDS.close();

    KBinaryDataStreamer bDS;
    bDS.open(file + std::string(".sbd"),"overwrite");
    bDS << surfaceContainer;
    bDS.close();

    KSADataStreamer saS;
    saS.open(file + std::string(".ksa"),"overwrite");
    saS << surfaceContainer;
    saS.close();

    KEMFileInterface::GetInstance()->Write(surfaceContainer,"surfaceContainer");
  }

#ifdef KEMFIELD_USE_PETSC
  KPETScInterface::GetInstance()->Finalize();
#elif KEMFIELD_USE_MPI
  KMPIInterface::GetInstance()->Finalize();
#endif

  return 0;
}
