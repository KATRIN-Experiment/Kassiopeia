#include <getopt.h>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

#include "KTypelist.hh"
#include "KSurfaceTypes.hh"
#include "KSurface.hh"
#include "KSurfaceContainer.hh"

#include "KDataDisplay.hh"

#include "KElectrostaticBoundaryIntegrator.hh"
#include "KBoundaryIntegralMatrix.hh"
#include "KBoundaryIntegralVector.hh"
#include "KBoundaryIntegralSolutionVector.hh"

#ifdef KEMFIELD_USE_VTK
#include "KEMVTKViewer.hh"
#endif

#include "KEMConstants.hh"

#include "KMPIInterface.hh"

#include "KPETScInterface.hh"
#include "KPETScSolver.hh"

#ifndef DEFAULT_DATA_DIR
#define DEFAULT_DATA_DIR "."
#endif /* !DEFAULT_DATA_DIR */

using namespace KEMField;

void ReadInTriangles(std::string fileName,KSurfaceContainer& surfaceContainer);
std::vector<std::string> Tokenize(std::string separators,std::string input);

void Field_Analytic(double Q,double radius1,double radius2,double radius3,double permittivity1,double permittivity2,double *P,double *F);

int main(int argc,char **argv)
{
  std::string usage =
    "\n"
    "Usage: TestPETSc <options>\n"
    "\n"
    "This program computes the charge densities of elements defined by input files.\n"
    "The program takes as inputs and outputs the names of geometry files to read.\n"
    "\n"
    "\tAvailable options:\n"
    "\t -h, --help               (shows this message and exits)\n"
    "\t -v, --verbose            (0..5; sets the verbosity)\n"
    "\t -s, --scale              (spherical capacitor scale)\n"
    "\n"
    "\t Additional PETSc options can be passed directly to the solver.\n";

  int verbose = 3;
  int scale = -1;

  static struct option longOptions[] = {
    {"help", no_argument, 0, 'h'},
    {"verbose", required_argument, 0, 'v'},
    {"scale", required_argument, 0, 's'},
  };

  static const char *optString = "hv:s:";

  opterr = 0;
  while(1) {
    char optId = getopt_long(argc, argv,optString, longOptions, NULL);
    if(optId == -1) break;
    switch(optId) {
    case('h'): // help
  if (KMPIInterface::GetInstance()->GetProcess()==0)
	std::cout<<usage<<std::endl;
      KPETScInterface::GetInstance()->Finalize();
      return 0;
    case('v'): // verbose
      verbose = atoi(optarg);
      if (verbose < 0) verbose = 0;
      if (verbose > 5) verbose = 5;
      break;
    case('s'):
      scale = atoi(optarg);
      if (scale < 1) scale = 1;
      if (scale >20) scale = 20;
      break;
    default:
      // do nothing
      break;
    }
  }

  KPETScInterface::GetInstance()->Initialize(&argc,&argv);

  if (scale == -1)
  {
    if (KMPIInterface::GetInstance()->GetProcess()==0)
      std::cout<<usage<<std::endl;
    KPETScInterface::GetInstance()->Finalize();
    return 0;
  }

  double radius1 = 1.;
  double radius2 = 2.;
  double radius3 = 3.;

  double potential1 = 1.;
  double permittivity1 = 2.;
  double permittivity2 = 3.;

  KSurfaceContainer surfaceContainer;

  std::stringstream s;
  s << DEFAULT_DATA_DIR << "/input_files/sphericalCapacitorFiles/SphericalCapacitor_" << scale << "_triangles.dat";
  KMPIInterface::GetInstance()->BeginSequentialProcess();
  ReadInTriangles(s.str(),surfaceContainer);
  KMPIInterface::GetInstance()->EndSequentialProcess();

  KElectrostaticBoundaryIntegrator integrator;
  KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator,KEM_USE_CACHING> A(surfaceContainer,integrator);
  KBoundaryIntegralSolutionVector<KElectrostaticBoundaryIntegrator> x(surfaceContainer,integrator);
  KBoundaryIntegralVector<KElectrostaticBoundaryIntegrator> b(surfaceContainer,integrator);

  KPETScSolver<KElectrostaticBoundaryIntegrator::ValueType> petscSolver;
  petscSolver.Solve(A,x,b);

#ifdef KEMFIELD_USE_VTK
  KEMVTKViewer viewer(surfaceContainer);
  viewer.GenerateGeometryFile("SphericalCapacitor.vtp");
  // viewer.ViewGeometry();
#endif

  double Q;
  double Q_b1;
  double Q_b2;
  double Q_b3;
  double Q_b4;
  double Q_computed;
  double Q_1;
  double Q_2;
  double Q_3;

  Q = potential1*4.*KEMConstants::Pi/
    (-1./(KEMConstants::Eps0*permittivity2*radius3) +
     1./(KEMConstants::Eps0*permittivity2*radius2) -
     1./(KEMConstants::Eps0*permittivity1*radius2) +
     1./(KEMConstants::Eps0*permittivity1*radius1));

  Q_b1 = -((permittivity1-1.)*Q/
	   (permittivity1));

  Q_b2 = ((permittivity1-1.)*Q/
	  (permittivity1));

  Q_b3 = -((permittivity2-1.)*Q/
	   (permittivity2));

  Q_b4 = ((permittivity2-1.)*Q/
	  (permittivity2));

  if (KMPIInterface::GetInstance()->GetProcess()==0)
  {
    std::cout<<"analytic charges: "<<std::endl;
    std::cout<<"Q: "<<Q<<std::endl;
    std::cout<<"Q_b1: "<<Q_b1<<std::endl;
    std::cout<<"Q_b2: "<<Q_b2<<std::endl;
    std::cout<<"Q_b3: "<<Q_b3<<std::endl;
    std::cout<<"Q_b4: "<<Q_b4<<std::endl;
    std::cout<<""<<std::endl;

    std::cout<<"analytic charges: "<<std::endl;
    std::cout<<"Q_1 analytic: "<<Q+Q_b1<<std::endl;
    std::cout<<"Q_2 analytic: "<<Q_b2+Q_b3<<std::endl;
    std::cout<<"Q_3 analytic: "<<Q_b4-Q<<std::endl;
    std::cout<<""<<std::endl;

    Q_computed = 0.;
    Q_1 = 0.;
    Q_2 = 0.;
    Q_3 = 0.;

    unsigned int i=0;
    for (KSurfaceContainer::iterator it=surfaceContainer.begin();
	 it!=surfaceContainer.end();it++)
    {
      if ((*it)->GetShape()->Centroid().Magnitude()<.5*(radius1+radius2))
	Q_1 += (*it)->GetShape()->Area() * x(i);
      else if ((*it)->GetShape()->Centroid().Magnitude()<.5*(radius3+radius2))
	Q_2 += (*it)->GetShape()->Area() * x(i);
      else
	Q_3 += (*it)->GetShape()->Area() * x(i);
      i++;
    }

    std::cout<<"total computed charge: "<<Q_computed<<std::endl;

    std::cout<<"Q_1: "<<Q_1<<std::endl;
    std::cout<<"Q_2: "<<Q_2<<std::endl;
    std::cout<<"Q_3: "<<Q_3<<std::endl;

    std::cout<<""<<std::endl;
    std::cout<<"comparisons:"<<std::endl;
    std::cout<<std::setprecision(16)<<"Q_1 vs (Q+Q_b1): "<<(Q_1-(Q+Q_b1))/(Q+Q_b1)*100.<<" %"<<std::endl;
    std::cout<<std::setprecision(16)<<"Q_2 vs (Q_b2+Q_b3): "<<(Q_2 - (Q_b2+Q_b3))/(Q_b2+Q_b3)*100.<<" %"<<std::endl;
    std::cout<<std::setprecision(16)<<"Q_3 vs (-Q+Q_b4): "<<(Q_3 - (-Q+Q_b4))/(-Q+Q_b4)*100.<<" %"<<std::endl;
  }

  KPETScInterface::GetInstance()->Finalize();
  return 0;
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

  bool dir_ = false;
  bool dir = false;
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
	// std::cout<<"reading in "<<token.at(0)<<" triangles"<<std::endl;
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
	  dir = true;
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
	  dir = false;
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
	dir_ = dir;
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

void Field_Analytic(double Q,double radius1,double radius2,double radius3,double permittivity1,double permittivity2,double *P,double *F)
{
  // This function computes the electric potential and electric field due to a
  // charge <Q> on a sphere of radius <radius1>, surrounded by two dielectrics.

  double r = sqrt(P[0]*P[0]+P[1]*P[1]+P[2]*P[2]);
  double fEps0 = 8.85418782e-12;

  if (r<radius1)
  {
    F[0] = Q/(4.*M_PI)*(-1./(fEps0*permittivity2*radius3) +
			1./(fEps0*permittivity2*radius2) -
			1./(fEps0*permittivity1*radius2) +
			1./(fEps0*permittivity1*radius1));
    F[1] = F[2] = F[3] = 0.;
  }
  else if (r<radius2)
  {
    F[0] = Q/(4.*M_PI)*(-1./(fEps0*permittivity2*radius3) +
			1./(fEps0*permittivity2*radius2) -
			1./(fEps0*permittivity1*radius2) +
			1./(fEps0*permittivity1*r));
    F[1] = Q/(4.*M_PI*fEps0*permittivity1*r*r)*P[0]/r;
    F[2] = Q/(4.*M_PI*fEps0*permittivity1*r*r)*P[1]/r;
    F[3] = Q/(4.*M_PI*fEps0*permittivity1*r*r)*P[2]/r;
  }
  else if (r<radius3)
  {
    F[0] = Q/(4.*M_PI)*(-1./(fEps0*permittivity2*radius3) +
			1./(fEps0*permittivity2*r));
    F[1] = Q/(4.*M_PI*fEps0*permittivity2*r*r)*P[0]/r;
    F[2] = Q/(4.*M_PI*fEps0*permittivity2*r*r)*P[1]/r;
    F[3] = Q/(4.*M_PI*fEps0*permittivity2*r*r)*P[2]/r;
  }
  else
  {
    F[0] = 0.;
    F[1] = 0.;
    F[2] = 0.;
    F[3] = 0.;
  }

}
