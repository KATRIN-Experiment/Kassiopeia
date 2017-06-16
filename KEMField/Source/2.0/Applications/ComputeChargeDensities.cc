#include <getopt.h>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <sys/stat.h>

#include "KElectrostaticBoundaryIntegratorFactory.hh"
#include "KTypelist.hh"
#include "KSurfaceTypes.hh"
#include "KSurface.hh"
#include "KSurfaceContainer.hh"

#include "KDataDisplay.hh"

#include "KSADataStreamer.hh"
#include "KBinaryDataStreamer.hh"
#include "KSerializer.hh"

#include "KIterativeStateReader.hh"
#include "KIterativeStateWriter.hh"

#include "KBoundaryIntegralMatrix.hh"
#include "KBoundaryIntegralVector.hh"
#include "KBoundaryIntegralSolutionVector.hh"

#include "KGaussianElimination.hh"
#include "KRobinHood.hh"

#ifdef KEMFIELD_USE_VTK
#include "KEMVTKViewer.hh"
#endif

#include "KEMConstants.hh"

#ifdef KEMFIELD_USE_MPI
#include "KRobinHood_MPI.hh"
#define MPI_SINGLE_PROCESS if (KMPIInterface::GetInstance()->GetProcess()==0)
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

#ifndef DEFAULT_DATA_DIR
#define DEFAULT_DATA_DIR "."
#endif /* !DEFAULT_DATA_DIR */

using namespace KEMField;

void ReadInTriangles(std::string fileName,KSurfaceContainer& surfaceContainer);
void ReadInRectangles(std::string fileName,KSurfaceContainer& surfaceContainer);
void ReadInWires(std::string fileName,KSurfaceContainer& surfaceContainer);
std::vector<std::string> Tokenize(std::string separators,std::string input);

void Field_Analytic(double Q,double radius1,double radius2,double radius3,double permittivity1,double permittivity2,double *P,double *F);

int main(int argc, char* argv[])
{
#ifdef KEMFIELD_USE_PETSC
  KPETScInterface::GetInstance()->Initialize(&argc,&argv);
#elif KEMFIELD_USE_MPI
  KMPIInterface::GetInstance()->Initialize(&argc,&argv);
#endif

  std::string usage =
    "\n"
    "Usage: ComputeChargeDensities <options>\n"
    "\n"
    "This program computes the charge densities of elements defined by input files.\n"
    "The program takes as inputs and outputs the names of geometry files to read.\n"
    "\n"
    "\tAvailable options:\n"
    "\t -h, --help               (shows this message and exits)\n"
    "\t -v, --verbose            (0..5; sets the verbosity)\n"
    "\t -f  --infile             (the name of the KEMField-style file)\n"
    "\t -t  --triangle-infile    (the name of the triangle file)\n"
    "\t -r  --rectangle-infile   (the name of the rectangle file)\n"
    "\t -w  --wire-infile        (the name of the wire file)\n"
    "\t -a, --accuracy           (accuracy of charge density computation)\n"
    "\t -i, --increment          (increment of accuracy check/print/log)\n"
    "\t -j, --save_increment     (increment of state saving)\n"
    "\t -k, --outfile            (name of the output file)\n"
    "\t -m, --method             (gauss"
#ifdef KEMFIELD_USE_PETSC
    ", robinhood or PETSc)\n";
#else
  " or robinhood)\n";
#endif

  int verbose = 3;
  double rh_accuracy = 1.e-8;
  int rh_increment = 100;
  int saveIncrement = UINT_MAX;
  int method = 1;

  std::string infile = "NULL";

  std::string triangleFile = "NULL";
  std::string rectangleFile = "NULL";
  std::string wireFile = "NULL";

  std::string outfile = "partialConvergence";

  static struct option longOptions[] = {
    {"help", no_argument, 0, 'h'},
    {"verbose", required_argument, 0, 'v'},
    {"infile", required_argument, 0, 'f'},
    {"triangle-infile", required_argument, 0, 't'},
    {"rectangle-infile", required_argument, 0, 'r'},
    {"wire-infile", required_argument, 0, 'w'},
    {"accuracy", required_argument, 0, 'a'},
    {"increment", required_argument, 0, 'i'},
    {"save_increment", required_argument, 0, 'j'},
    {"outfile", required_argument, 0, 'k'},
    {"method", required_argument, 0, 'm'},
  };

  static const char *optString = "hv:f:t:r:w:a:g:i:j:k:m:";

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
      if (verbose < 0) verbose = 0;
      if (verbose > 5) verbose = 5;
      break;
    case('a'):
      rh_accuracy = atof(optarg);
      break;
    case('f'):
      infile = optarg;
      break;
    case('t'):
      triangleFile = optarg;
      break;
    case('r'):
      rectangleFile = optarg;
      break;
    case('w'):
      wireFile = optarg;
      break;
    case('i'):
      rh_increment = atoi(optarg);
      break;
    case('j'):
      saveIncrement = atoi(optarg);
      break;
    case('k'):
      outfile = optarg;
      break;
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

  KSurfaceContainer surfaceContainer;

  if (infile != "NULL")
  {
#ifdef KEMFIELD_USE_MPI
    KMPIInterface::GetInstance()->BeginSequentialProcess();
#endif
    std::string suffix = infile.substr(infile.find_last_of("."),
				       std::string::npos);

    KSADataStreamer ksaDataStreamer;
    KBinaryDataStreamer binaryDataStreamer;

    struct stat fileInfo;
    bool exists;
    int fileStat;

    // Attempt to get the file attributes
    fileStat = stat(infile.c_str(),&fileInfo);
    if(fileStat == 0)
      exists = true;
    else
      exists = false;

    if (!exists)
    {
      std::cout<<"Error: file \""<<infile<<"\" cannot be read."<<std::endl;
      return 1;
    }

    if (suffix.compare(ksaDataStreamer.GetFileSuffix()) == 0)
    {
      ksaDataStreamer.open(infile,"read");
      ksaDataStreamer >> surfaceContainer;
      ksaDataStreamer.close();
    }
    else if (suffix.compare(binaryDataStreamer.GetFileSuffix()) == 0)
    {
      binaryDataStreamer.open(infile,"read");
      binaryDataStreamer >> surfaceContainer;
      binaryDataStreamer.close();
    }
    else
    {
      std::cout<<"Error: unkown file extension \""<<suffix<<"\""<<std::endl;
    }

#ifdef KEMFIELD_USE_MPI
    KMPIInterface::GetInstance()->EndSequentialProcess();
#endif
  }

  if (triangleFile != "NULL")
  {
#ifdef KEMFIELD_USE_MPI
    KMPIInterface::GetInstance()->BeginSequentialProcess();
#endif
    ReadInTriangles(triangleFile,surfaceContainer);
#ifdef KEMFIELD_USE_MPI
    KMPIInterface::GetInstance()->EndSequentialProcess();
#endif
  }
  if (rectangleFile != "NULL")
  {
#ifdef KEMFIELD_USE_MPI
    KMPIInterface::GetInstance()->BeginSequentialProcess();
#endif
    ReadInRectangles(rectangleFile,surfaceContainer);
#ifdef KEMFIELD_USE_MPI
    KMPIInterface::GetInstance()->EndSequentialProcess();
#endif
  }
  if (wireFile != "NULL")
  {
#ifdef KEMFIELD_USE_MPI
    KMPIInterface::GetInstance()->BeginSequentialProcess();
#endif
    ReadInWires(wireFile,surfaceContainer);
#ifdef KEMFIELD_USE_MPI
    KMPIInterface::GetInstance()->EndSequentialProcess();
#endif
  }

  if (surfaceContainer.empty())
  {
    MPI_SINGLE_PROCESS
    {
      std::cout<<"Error: surface container is empty."<<std::endl;
      std::cout<<usage<<std::endl;
    }
#ifdef KEMFIELD_USE_MPI
    KMPIInterface::GetInstance()->Finalize();
#endif
    return 1;
  }

#ifdef KEMFIELD_USE_MPI
#ifdef KEMFIELD_USE_OPENCL
  KOpenCLInterface::GetInstance()->SetGPU(KMPIInterface::GetInstance()->GetProcess());
#endif
#endif

  // let's see if we can find this geometry
  bool skipBEM = false;
  {
    KMD5HashGenerator hashGenerator;
    hashGenerator.Omit(KBasisTypes());

    std::vector<std::string> labels;

    labels.push_back(hashGenerator.GenerateHash(surfaceContainer));
    labels.push_back("residual_threshold");

    unsigned int nElements = KEMFileInterface::GetInstance()->NumberWithLabels( labels );

    if (nElements>0)
    {
      KResidualThreshold residualThreshold;
      KResidualThreshold minResidualThreshold;

      for (unsigned int i=0;i<nElements;i++)
      {
	KEMFileInterface::GetInstance()->FindByLabels(residualThreshold,labels,i);
	if (residualThreshold<minResidualThreshold)
	  minResidualThreshold = residualThreshold;
      }

      KEMFileInterface::GetInstance()->FindByHash(surfaceContainer,minResidualThreshold.fGeometryHash);

      if (minResidualThreshold.fResidualThreshold<=rh_accuracy)
	skipBEM = true;
    }
  }

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

  if (!skipBEM)
  {
    if (method == 0)
    {
      KGaussianElimination<KElectrostaticBoundaryIntegrator::ValueType> gaussianElimination;
      gaussianElimination.Solve(A,x,b);
    }
    else if (method == 1)
    {
#if defined(KEMFIELD_USE_MPI) && defined(KEMFIELD_USE_OPENCL)
      KRobinHood<KElectrostaticBoundaryIntegrator::ValueType,
		 KRobinHood_MPI_OpenCL> robinHood;
#elif defined(KEMFIELD_USE_MPI)
      KRobinHood<KElectrostaticBoundaryIntegrator::ValueType,
		 KRobinHood_MPI> robinHood;
#elif defined(KEMFIELD_USE_OPENCL)
      KRobinHood<KElectrostaticBoundaryIntegrator::ValueType,
		 KRobinHood_OpenCL> robinHood;
#else
      KRobinHood<KElectrostaticBoundaryIntegrator::ValueType> robinHood;
#endif

      robinHood.SetTolerance(rh_accuracy);

#ifdef KEMFIELD_USE_OPENCL
#ifndef KEMFIELD_USE_DOUBLE_PRECISION
      robinHood.SetTolerance((rh_accuracy > 1.e-5 ? rh_accuracy : 1.e-5));
#endif
#endif

      outfile = outfile.substr(0,outfile.find_last_of("."));

      KIterativeStateReader<KElectrostaticBoundaryIntegrator::ValueType>* stateReader = new KIterativeStateReader<KElectrostaticBoundaryIntegrator::ValueType>(surfaceContainer);
      robinHood.AddVisitor(stateReader);

      MPI_SINGLE_PROCESS
      {
	robinHood.AddVisitor(new KIterationDisplay<KElectrostaticBoundaryIntegrator::ValueType>());
      }

      KIterativeStateWriter<KElectrostaticBoundaryIntegrator::ValueType>* stateWriter = new KIterativeStateWriter<KElectrostaticBoundaryIntegrator::ValueType>(surfaceContainer);
      stateWriter->Interval(saveIncrement);
      stateWriter->SaveNameRoot(outfile);
      robinHood.AddVisitor(stateWriter);

      robinHood.SetResidualCheckInterval(rh_increment);
      robinHood.Solve(A,x,b);
    }
#ifdef KEMFIELD_USE_PETSC
    else if (method == 2)
    {
      KPETScSolver<KElectrostaticBoundaryIntegrator::ValueType> petscSolver;
      petscSolver.Solve(A,x,b);
    }
#endif
  }

#ifdef KEMFIELD_USE_VTK
  KEMVTKViewer viewer(surfaceContainer);
  viewer.GenerateGeometryFile("electrodes.vtp");
  // viewer.ViewGeometry();
#endif

  MPI_SINGLE_PROCESS
  {
    KMetadataStreamer mDS;
    mDS.open(outfile + std::string(".smd"),"overwrite");
    mDS << surfaceContainer;
    mDS.close();

    KBinaryDataStreamer bDS;
    bDS.open(outfile + std::string(".kbd"),"overwrite");
    bDS << surfaceContainer;
    bDS.close();

    KSADataStreamer saS;
    saS.open(outfile + std::string(".ksa"),"overwrite");
    saS << surfaceContainer;
    saS.close();
  }

  // KSADataStreamer csaS;
  // csaS.open("electrodes.zksa","overwrite");
  // csaS << surfaceContainer;
  // csaS.close();


#ifdef KEMFIELD_USE_PETSC
  KPETScInterface::GetInstance()->Finalize();
#elif KEMFIELD_USE_MPI
  KMPIInterface::GetInstance()->Finalize();
#endif
}

void ReadInTriangles(std::string fileName,KSurfaceContainer& surfaceContainer)
{
  typedef KSurface<KElectrostaticBasis,KDirichletBoundary,KTriangle> KDirichletTriangle;
  typedef KSurface<KElectrostaticBasis,KNeumannBoundary,KTriangle> KNeumannTriangle;

  std::string inBuf;
  std::vector<std::string> token;
  std::fstream file;

  file.open(fileName.c_str(), std::fstream::in);

  getline(file,inBuf);
  token = Tokenize(" \t",inBuf);

  int lineNum = 0;

//  bool dir_ = false; // TODO: WHAT ARE THESE VARIABLES FOR?
//  bool dir = false;
  int counter = 0;

  while (!file.eof())
  {
    lineNum++;
    if (token.size()>0)
    {
      if (token.at(0).at(0)=='#')
      {
	getline(file,inBuf);
	token = Tokenize(" \t",inBuf);
	continue;
      }
    }

    if (token.size()!=0)
    {
      if (token.size()==1)
      {
	MPI_SINGLE_PROCESS
	  KEMField::cout<<"Reading in "<<token.at(0)<<" triangles."<<KEMField::endl;
      }
      else
      {
	std::stringstream s;
	int n[4];
	double d[15];

	for (int i=0;i<4;i++)
	{ s.str(token.at(i)); s >> n[i]; s.clear(); }
	for (int i=0;i<15;i++)
	{ s.str(token.at(i+4)); s >> d[i]; s.clear(); }

	if (fabs(d[13]/d[12])<1.e-10)
	{
//	  dir = true;   // TODO: WHAT ARE THESE VARIABLES FOR?
	  // if (dir_ == false)
	  //   std::cout<<"switch to dirichlet at "<<counter<<std::endl;

	  KDirichletTriangle* t = new KDirichletTriangle();

	  t->SetA(d[9]);
	  t->SetB(d[10]);
	  t->SetP0(KPosition(d[0],d[1],d[2]));
	  t->SetN1(KPosition(d[3],d[4],d[5]));
	  t->SetN2(KPosition(d[6],d[7],d[8]));
	  t->SetBoundaryValue(d[11]);

	  surfaceContainer.push_back(t);
	}
	else
	{
//	  dir = false;  // TODO: WHAT ARE THESE VARIABLES FOR?
	  // if (dir_ == true)
	  //   std::cout<<"switch to neumann at "<<counter<<std::endl;

	  KNeumannTriangle* t = new KNeumannTriangle();

	  t->SetA(d[9]);
	  t->SetB(d[10]);
	  t->SetP0(KPosition(d[0],d[1],d[2]));
	  t->SetN1(KPosition(d[3],d[4],d[5]));
	  t->SetN2(KPosition(d[6],d[7],d[8]));
	  t->SetNormalBoundaryFlux(d[13]/d[12]);

	  surfaceContainer.push_back(t);
	}
//	dir_ = dir;
	counter++;
      }
    }
    getline(file,inBuf);
    token = Tokenize(" \t",inBuf);
  }
  file.close();
}

void ReadInRectangles(std::string fileName,KSurfaceContainer& surfaceContainer)
{
  typedef KSurface<KElectrostaticBasis,KDirichletBoundary,KRectangle> KDirichletRectangle;
  typedef KSurface<KElectrostaticBasis,KNeumannBoundary,KRectangle> KNeumannRectangle;

  std::string inBuf;
  std::vector<std::string> token;
  std::fstream file;

  file.open(fileName.c_str(), std::fstream::in);

  getline(file,inBuf);
  token = Tokenize(" \t",inBuf);

  int lineNum = 0;

//  bool dir_ = false; // TODO: WHAT ARE THESE VARIABLES FOR?
//  bool dir = false;
  int counter = 0;

  while (!file.eof())
  {
    lineNum++;
    if (token.size()>0)
    {
      if (token.at(0).at(0)=='#')
      {
	getline(file,inBuf);
	token = Tokenize(" \t",inBuf);
	continue;
      }
    }

    if (token.size()!=0)
    {
      if (token.size()==1) std::cout<<"reading in "<<token.at(0)<<" rectangles"<<std::endl;
      else
      {
	std::stringstream s;
	int n[4];
	double d[15];

	for (int i=0;i<4;i++)
	{ s.str(token.at(i)); s >> n[i]; s.clear(); }
	for (int i=0;i<15;i++)
	{ s.str(token.at(i+4)); s >> d[i]; s.clear(); }

	if (fabs(d[13]/d[12])<1.e-10)
	{
//	  dir = true; // TODO: WHAT ARE THESE VARIABLES FOR?
	  // if (dir_ == false)
	  //   std::cout<<"switch to dirichlet at "<<counter<<std::endl;

	  KDirichletRectangle* t = new KDirichletRectangle();

	  t->SetA(d[9]);
	  t->SetB(d[10]);
	  t->SetP0(KPosition(d[0],d[1],d[2]));
	  t->SetN1(KPosition(d[3],d[4],d[5]));
	  t->SetN2(KPosition(d[6],d[7],d[8]));
	  t->SetBoundaryValue(d[11]);

	  surfaceContainer.push_back(t);
	}
	else
	{
//	  dir = false; // TODO: WHAT ARE THESE VARIABLES FOR?
	  // if (dir_ == true)
	  //   std::cout<<"switch to neumann at "<<counter<<std::endl;

	  KNeumannRectangle* t = new KNeumannRectangle();

	  t->SetA(d[9]);
	  t->SetB(d[10]);
	  t->SetP0(KPosition(d[0],d[1],d[2]));
	  t->SetN1(KPosition(d[3],d[4],d[5]));
	  t->SetN2(KPosition(d[6],d[7],d[8]));
	  t->SetNormalBoundaryFlux(d[13]/d[12]);

	  surfaceContainer.push_back(t);
	}
//	dir_ = dir; // TODO: WHAT ARE THESE VARIABLES FOR?
	counter++;
      }
    }
    getline(file,inBuf);
    token = Tokenize(" \t",inBuf);
  }
  file.close();
}

void ReadInWires(std::string fileName,KSurfaceContainer& surfaceContainer)
{
  typedef KSurface<KElectrostaticBasis,KDirichletBoundary,KLineSegment> KDirichletWire;

  std::string inBuf;
  std::vector<std::string> token;
  std::fstream file;

  file.open(fileName.c_str(), std::fstream::in);

  getline(file,inBuf);
  token = Tokenize(" \t",inBuf);

  int lineNum = 0;

//  bool dir_ = false;
//  bool dir = false;
  int counter = 0;

  while (!file.eof())
  {
    lineNum++;
    if (token.size()>0)
    {
      if (token.at(0).at(0)=='#')
      {
	getline(file,inBuf);
	token = Tokenize(" \t",inBuf);
	continue;
      }
    }

    if (token.size()!=0)
    {
      if (token.size()==1) std::cout<<"reading in "<<token.at(0)<<" wires"<<std::endl;
      else
      {
	std::stringstream s;
	int n[4];
	double d[15];

	for (int i=0;i<4;i++)
	{ s.str(token.at(i)); s >> n[i]; s.clear(); }
	for (int i=0;i<15;i++)
	{ s.str(token.at(i+4)); s >> d[i]; s.clear(); }

	if (fabs(d[13]/d[12])<1.e-10)
	{
//	  dir = true; // TODO: WHAT ARE THESE VARIABLES FOR?
	  // if (dir_ == false)
	  //   std::cout<<"switch to dirichlet at "<<counter<<std::endl;

	  KDirichletWire* t = new KDirichletWire();

	  t->SetP0(KPosition(d[0],d[1],d[2]));
	  t->SetP1(KPosition(d[3],d[4],d[5]));
	  t->SetDiameter(d[6]);
	  t->SetBoundaryValue(d[11]);

	  surfaceContainer.push_back(t);
	}
//	dir_ = dir; // TODO: WHAT ARE THESE VARIABLES FOR?
	counter++;
      }
    }
    getline(file,inBuf);
    token = Tokenize(" \t",inBuf);
  }
  file.close();
}

std::vector<std::string> Tokenize(std::string separators,std::string input)
{
  unsigned int startToken = 0, endToken; // Pointers to the token pos
  std::vector<std::string> tokens;       // Vector to keep the tokens
  unsigned int commentPos = input.size()+1;

  if( separators.size() > 0 && input.size() > 0 )
  {
    // Check for comment
    for (unsigned int i=0;i<input.size();i++)
    {
      if (input[i]=='#' || (i<input.size()-1&&(input[i]=='/'&&input[i+1]=='/')))
      {
	commentPos=i;
	break;
      }
    }

    while( startToken < input.size() )
    {
      // Find the start of token
      startToken = input.find_first_not_of( separators, startToken );

      // Stop parsing when comment symbol is reached
      if (startToken == commentPos)
      {
	if (tokens.size()==0)
	  tokens.push_back("#");
	return tokens;
      }

      // If found...
      if( startToken != (unsigned int)std::string::npos)
      {
	// Find end of token
	endToken = input.find_first_of( separators, startToken );

	if( endToken == (unsigned int)std::string::npos )
	  // If there was no end of token, assign it to the end of string
	  endToken = input.size();

	// Extract token
	tokens.push_back( input.substr( startToken, endToken - startToken ) );

	// Update startToken
	startToken = endToken;
      }
    }
  }

  return tokens;
}
