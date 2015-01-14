#include <getopt.h>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>

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

#include "KEMFieldCanvas.hh"

#ifdef KEMFIELD_USE_VTK
#include "KEMVTKViewer.hh"
#include "KVTKIterationPlotter.hh"
#include "KEMVTKFieldCanvas.hh"
#endif

#include "KEMConstants.hh"

#include "KBiconjugateGradientStabilized.hh"
#include "KBiconjugateGradientStabilizedJacobiPreconditioned_SingleThread.hh"
#include "KGeneralizedMinimalResidual.hh"
#include "KGeneralizedMinimalResidual_SingleThread.hh"

#include "KFMBoundaryIntegralMatrix.hh"
#include "KFMElectrostaticBoundaryIntegrator.hh"
#include "KFMElectrostaticFastMultipoleFieldSolver.hh"

#ifdef KEMFIELD_USE_OPENCL
#include "KOpenCLSurfaceContainer.hh"
#include "KOpenCLElectrostaticBoundaryIntegrator.hh"
#include "KOpenCLBoundaryIntegralMatrix.hh"
#include "KOpenCLBoundaryIntegralVector.hh"
#include "KOpenCLBoundaryIntegralSolutionVector.hh"
#include "KRobinHood_OpenCL.hh"
#include "KFMElectrostaticTreeManager_OpenCL.hh"
#endif

#ifndef DEFAULT_OUTPUT_DIR
#define DEFAULT_OUTPUT_DIR "."
#endif /* !DEFAULT_OUTPUT_DIR */

using namespace KGeoBag;
using namespace KEMField;


////////////////////////////////////////////////////////////////////////////////
//Configuration class for this test program

namespace KEMField{

class ConfigureTestFFTMSolver_SphereGMSH: public KSAInputOutputObject
{
    public:

        ConfigureTestFFTMSolver_SphereGMSH()
        {
            fOutputFileName = std::string("");
            fNSamplePoints = 1;
            fGMSHScale = 0;
            fDivisions = 0;
            fDegree = 0;
            fZeroMaskSize = 0;
            fMaxTreeLevel = 0;
            fRegionExpansionFactor = 1;
            fRegionToPrimitiveSizeRatio = 1;
            fAggressionLevel = 0;
            fMaxDesiredNumberOfOwnedPrimitives = 1;
        }

        virtual ~ConfigureTestFFTMSolver_SphereGMSH(){;};

        virtual const char* GetName() const {return "ConfigureTestFFTMSolver_SphereGMSH"; };

        std::string GetOutputFileName() const {return fOutputFileName;};
        void SetOutputFileName(const std::string& name){fOutputFileName = name;};

        UInt_t GetNSamplePoints() const {return fNSamplePoints;};
        void SetNSamplePoints(const UInt_t& n){fNSamplePoints = n;};

        Int_t GetGMSHScale() const {return fGMSHScale;};
        void SetGMSHScale(const Int_t& s){fGMSHScale = s;};

        Int_t GetDivisions() const {return fDivisions;};
        void SetDivisions(const Int_t& d){fDivisions = d;};

        Int_t GetDegree() const {return fDegree;};
        void SetDegree(const Int_t& deg){fDegree = deg;};

        Int_t GetZeroMaskSize() const {return fZeroMaskSize;};
        void SetZeroMaskSize(const Int_t& z){fZeroMaskSize = z;};

        Int_t GetMaxTreeLevel() const {return fMaxTreeLevel;};
        void SetMaxTreeLevel(const Int_t& t){fMaxTreeLevel = t;};

        Int_t GetMaxDesiredNumberOfOwnedPrimitives() const {return fMaxDesiredNumberOfOwnedPrimitives;};
        void SetMaxDesiredNumberOfOwnedPrimitives(const Int_t& md){fMaxDesiredNumberOfOwnedPrimitives = md;};

        Double_t GetRegionExpansionFactor() const {return fRegionExpansionFactor;};
        void SetRegionExpansionFactor(const Double_t& f){fRegionExpansionFactor = f;};

        Double_t GetRegionToPrimitiveSizeRatio() const {return fRegionToPrimitiveSizeRatio;};
        void SetRegionToPrimitiveSizeRatio(const Double_t& rho){fRegionToPrimitiveSizeRatio = rho;};

        Int_t GetAggressionLevel() const {return fAggressionLevel;};
        void SetAggressionLevel(const Int_t& agg){fAggressionLevel = agg;};

        void DefineOutputNode(KSAOutputNode* node) const
        {
            AddKSAOutputFor(ConfigureTestFFTMSolver_SphereGMSH, OutputFileName, std::string);
            AddKSAOutputFor(ConfigureTestFFTMSolver_SphereGMSH,NSamplePoints,UInt_t);
            AddKSAOutputFor(ConfigureTestFFTMSolver_SphereGMSH,GMSHScale,Int_t);
            AddKSAOutputFor(ConfigureTestFFTMSolver_SphereGMSH,Divisions,Int_t);
            AddKSAOutputFor(ConfigureTestFFTMSolver_SphereGMSH,Degree,Int_t);
            AddKSAOutputFor(ConfigureTestFFTMSolver_SphereGMSH,ZeroMaskSize,Int_t);
            AddKSAOutputFor(ConfigureTestFFTMSolver_SphereGMSH,MaxTreeLevel,Int_t);
            AddKSAOutputFor(ConfigureTestFFTMSolver_SphereGMSH,MaxDesiredNumberOfOwnedPrimitives, Int_t);
            AddKSAOutputFor(ConfigureTestFFTMSolver_SphereGMSH,RegionExpansionFactor,Double_t);
            AddKSAOutputFor(ConfigureTestFFTMSolver_SphereGMSH,RegionToPrimitiveSizeRatio,Double_t);
            AddKSAOutputFor(ConfigureTestFFTMSolver_SphereGMSH,AggressionLevel,Int_t);
        }

        void DefineInputNode(KSAInputNode* node)
        {
            AddKSAInputFor(ConfigureTestFFTMSolver_SphereGMSH, OutputFileName, std::string);
            AddKSAInputFor(ConfigureTestFFTMSolver_SphereGMSH,NSamplePoints,UInt_t);
            AddKSAInputFor(ConfigureTestFFTMSolver_SphereGMSH,GMSHScale,Int_t);
            AddKSAInputFor(ConfigureTestFFTMSolver_SphereGMSH,Divisions,Int_t);
            AddKSAInputFor(ConfigureTestFFTMSolver_SphereGMSH,Degree,Int_t);
            AddKSAInputFor(ConfigureTestFFTMSolver_SphereGMSH,ZeroMaskSize,Int_t);
            AddKSAInputFor(ConfigureTestFFTMSolver_SphereGMSH,MaxTreeLevel,Int_t);
            AddKSAInputFor(ConfigureTestFFTMSolver_SphereGMSH,MaxDesiredNumberOfOwnedPrimitives, Int_t);
            AddKSAInputFor(ConfigureTestFFTMSolver_SphereGMSH,RegionExpansionFactor,Double_t);
            AddKSAInputFor(ConfigureTestFFTMSolver_SphereGMSH,RegionToPrimitiveSizeRatio,Double_t);
            AddKSAInputFor(ConfigureTestFFTMSolver_SphereGMSH,AggressionLevel,Int_t);
        }

        virtual const char* ClassName() const { return "ConfigureTestFFTMSolver_SphereGMSH"; };

    protected:

        std::string fOutputFileName;
        Int_t fGMSHScale;
        Int_t fDivisions;
        Int_t fDegree;
        Int_t fZeroMaskSize;
        Int_t fMaxTreeLevel;
        Int_t fMaxDesiredNumberOfOwnedPrimitives;
        Double_t fRegionExpansionFactor;
        Double_t fRegionToPrimitiveSizeRatio;
        Int_t fAggressionLevel;
        UInt_t fNSamplePoints;

};

DefineKSAClassName( ConfigureTestFFTMSolver_SphereGMSH );

}


////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
  std::string usage =
    "\n"
    "Usage: TestFastMultipoleSphereCapacitance <options>\n"
    "\n"
    "This program computes the capacitance of the unit sphere, and compares them to\n"
    "the analytic value.\n"
    "\n"
    "\tAvailable options:\n"
    "\t -h, --help               (shows this message and exits)\n"
    "\t -v, --verbose            (0..5; sets the verbosity)\n"
    "\t -s, --scale              (discretization scale)\n"
    "\t -a, --accuracy           (accuracy of charge density computation)\n"
    "\t -i, --increment          (increment of accuracy check/print/log)\n"
    "\t -d, --divisions          (number of divisions per dimension in tree branching)\n"
    "\t -p, --degree             (degree of the multipole expansion)\n"
    "\t -z, --zero-mask          (size of the zeromask in the response functions)\n"
    "\t -t, --tree-depth         (maximum depth of the region tree)\n"
#ifdef KEMFIELD_USE_VTK
    "\t -e, --with-plot          (dynamic plot of residual norm)\n"
#endif
    "\t -m, --method             ( (1) biCGSTAB, (2) Jacobi-biCGSTAB, (3) GMRES )\n";

  int verbose = 3;
  int scale = 1;
  double accuracy = 1.e-5;
  int increment = 100;
  bool usePlot;
  usePlot = false;
  int method = 3;
  int gmres_restart = 100;

  int divisions = 3;
  int degree = 4;
  int zeromask = 1;
  int max_tree_depth = 4;

  static struct option longOptions[] = {
    {"help", no_argument, 0, 'h'},
    {"verbose", required_argument, 0, 'v'},
    {"scale", required_argument, 0, 's'},
    {"accuracy", required_argument, 0, 'a'},
    {"increment", required_argument, 0, 'i'},
    {"divisions", required_argument, 0, 'd'},
    {"degree", required_argument, 0, 'p'},
    {"zero-mask", required_argument, 0, 'z'},
    {"tree-depth", required_argument, 0, 't'},
#ifdef KEMFIELD_USE_VTK
    {"with-plot", no_argument, 0, 'e'},
#endif
    {"method", required_argument, 0, 'm'}
  };

#ifdef KEMFIELD_USE_VTK
  static const char *optString = "hv:s:a:i:d:p:z:t:em:";
#else
  static const char *optString = "hv:s:a:i:d:p:z:t:m:";
#endif

  while(1) {
    char optId = getopt_long(argc, argv,optString, longOptions, NULL);
    if(optId == -1) break;
    switch(optId) {
    case('h'): // help
      std::cout<<usage<<std::endl;
      return 0;
    case('v'): // verbose
      verbose = atoi(optarg);
      if (verbose < 0) verbose = 0;
      if (verbose > 5) verbose = 5;
      break;
    case('s'):
      scale = atoi(optarg);
      break;
    case('a'):
      accuracy = atof(optarg);
      break;
    case('i'):
      increment = atoi(optarg);
      break;
    case('d'):
      divisions = atoi(optarg);
      break;
    case('p'):
      degree = atoi(optarg);
      break;
    case('z'):
      zeromask = atoi(optarg);
      break;
    case('t'):
      max_tree_depth = atoi(optarg);
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
      std::cout<<usage<<std::endl;
      return 1;
    }
  }

  if (scale < 1)
  {
      std::cout<<usage<<std::endl;
      return 1;
  }

  KEMField::cout.Verbose(false);
  KEMField::cout.Verbose(verbose);

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

  KSurfaceContainer surfaceContainer;

  KGBEMMeshConverter geometryConverter(surfaceContainer);
  geometryConverter.SetMinimumArea(1.e-12);
  hemisphere1->AcceptNode(&geometryConverter);
  hemisphere2->AcceptNode(&geometryConverter);

  KEMField::cout<<"Computing the capacitance for a unit sphere comprised of "<<surfaceContainer.size()<<" elements"<<KEMField::endl;

    KFMElectrostaticParameters params;
    params.divisions = divisions;
    params.degree = degree;
    params.zeromask = zeromask;
    params.maximum_tree_depth = max_tree_depth;
    params.region_expansion_factor = 1.1;

#ifndef KEMFIELD_USE_OPENCL

    typedef KFMElectrostaticBoundaryIntegrator<KFMElectrostaticTreeManager_SingleThread> KFMSingleThreadEBI;
    KFMSingleThreadEBI integrator(surfaceContainer);
    integrator.Initialize(params);
    KFMBoundaryIntegralMatrix< KFMSingleThreadEBI > A(surfaceContainer,integrator);
    KBoundaryIntegralSolutionVector< KFMSingleThreadEBI > x(surfaceContainer,integrator);
    KBoundaryIntegralVector< KFMSingleThreadEBI > b(surfaceContainer,integrator);

    if(method == 1)
    {

        KBiconjugateGradientStabilized< KFMSingleThreadEBI::ValueType, KBiconjugateGradientStabilized_SingleThread> biCGSTAB;
        biCGSTAB.SetTolerance(accuracy);

        biCGSTAB.AddVisitor(new KIterationDisplay<double>());

    #ifdef KEMFIELD_USE_VTK
        if(usePlot)
        {
            biCGSTAB.AddVisitor(new KVTKIterationPlotter<double>());
        }
    #endif

        biCGSTAB.Solve(A,x,b);

    }
    else if(method == 2)
    {

        KBiconjugateGradientStabilized< KFMSingleThreadEBI::ValueType, KBiconjugateGradientStabilizedJacobiPreconditioned_SingleThread> biCGSTAB;
        biCGSTAB.SetTolerance(accuracy);

        biCGSTAB.AddVisitor(new KIterationDisplay<double>());

    #ifdef KEMFIELD_USE_VTK
        if(usePlot)
        {
            biCGSTAB.AddVisitor(new KVTKIterationPlotter<double>());
        }
    #endif

        biCGSTAB.Solve(A,x,b);
    }
    else if(method == 3)
    {
        KGeneralizedMinimalResidual< KFMSingleThreadEBI::ValueType, KGeneralizedMinimalResidual_SingleThread> gmres;
        gmres.SetTolerance(accuracy);
        gmres.SetRestartParameter(gmres_restart);

        gmres.AddVisitor(new KIterationDisplay<double>());

    #ifdef KEMFIELD_USE_VTK
        if(usePlot)
        {
            gmres.AddVisitor(new KVTKIterationPlotter<double>());
        }
    #endif

        gmres.Solve(A,x,b);
    }

#else


    typedef KFMElectrostaticBoundaryIntegrator<KFMElectrostaticTreeManager_OpenCL> KFMOpenCLEBI;
    KFMOpenCLEBI integrator(surfaceContainer);
    integrator.Initialize(params);
    KFMBoundaryIntegralMatrix< KFMOpenCLEBI > A(surfaceContainer,integrator);
    KBoundaryIntegralSolutionVector<KFMOpenCLEBI > x(surfaceContainer,integrator);
    KBoundaryIntegralVector< KFMOpenCLEBI> b(surfaceContainer,integrator);

    if(method == 1)
    {

        KBiconjugateGradientStabilized< KFMOpenCLEBI::ValueType, KBiconjugateGradientStabilized_SingleThread> biCGSTAB;
        biCGSTAB.SetTolerance(accuracy);


        biCGSTAB.AddVisitor(new KIterationDisplay<double>());

    #ifdef KEMFIELD_USE_VTK
        if(usePlot)
        {
            biCGSTAB.AddVisitor(new KVTKIterationPlotter<double>());
        }
    #endif

        biCGSTAB.Solve(A,x,b);

    }
    else if(method == 2)
    {

        KBiconjugateGradientStabilized< KFMOpenCLEBI::ValueType, KBiconjugateGradientStabilizedJacobiPreconditioned_SingleThread> biCGSTAB;
        biCGSTAB.SetTolerance(accuracy);


        biCGSTAB.AddVisitor(new KIterationDisplay<double>());

    #ifdef KEMFIELD_USE_VTK
        if(usePlot)
        {
            biCGSTAB.AddVisitor(new KVTKIterationPlotter<double>());
        }
    #endif

        biCGSTAB.Solve(A,x,b);
    }
    else if(method == 3)
    {
        KGeneralizedMinimalResidual< KFMOpenCLEBI::ValueType, KGeneralizedMinimalResidual_SingleThread> gmres;
        gmres.SetTolerance(accuracy);
        gmres.SetRestartParameter(gmres_restart);

        gmres.AddVisitor(new KIterationDisplay<double>());

    #ifdef KEMFIELD_USE_VTK
        if(usePlot)
        {
            gmres.AddVisitor(new KVTKIterationPlotter<double>());
        }
    #endif

        gmres.Solve(A,x,b);
    }

#endif


//  if (usePlot)
//  {
//    KElectrostaticBoundaryIntegrator integrator;
//    KBoundaryIntegralMatrix<KElectrostaticBoundaryIntegrator> A(surfaceContainer,integrator);

//    KEMFieldCanvas* fieldCanvas = NULL;

//#if defined(KEMFIELD_USE_VTK)
//    fieldCanvas = new KEMVTKFieldCanvas(0.,double(A.Dimension()),0.,double(A.Dimension()),1.e30,true);
//#endif

//    if (fieldCanvas)
//    {
//      std::vector<double> x_;
//      std::vector<double> y_;
//      std::vector<double> V_;

//      for (unsigned int i=0;i<A.Dimension();i++)
//      {
//	x_.push_back(i);
//	y_.push_back(i);

//	for (unsigned int j=0;j<A.Dimension();j++)
//	{
//	  double value = A(i,j);
//	  if (value > 1.e-16)
//	    V_.push_back(log(value));
//	  else
//	    V_.push_back(-16.);
//	}
//      }

//      fieldCanvas->DrawFieldMap(x_,y_,V_,false,0.);
//      fieldCanvas->LabelAxes("i","j","log (A_{ij})");
//      std::string fieldCanvasName = DEFAULT_OUTPUT_DIR;
//      fieldCanvas->SaveAs(fieldCanvasName + "/Matrix.gif");
//    }
//  }


    double Q = 0.;

    unsigned int i=0;
    for (KSurfaceContainer::iterator it=surfaceContainer.begin();
	 it!=surfaceContainer.end();it++)
    {
      Q += (dynamic_cast<KTriangle*>(*it)->Area() *
	    dynamic_cast<KElectrostaticBasis*>(*it)->GetSolution());
      i++;
    }

    std::cout<<""<<std::endl;
    double C = Q/(4.*M_PI*KEMConstants::Eps0);

    double C_Analytic = 1.;

    std::cout<<std::setprecision(7)<<"Capacitance:    "<<C<<std::endl;
    std::cout.setf( std::ios::fixed, std:: ios::floatfield );
    std::cout<<std::setprecision(7)<<"Accepted value: "<<C_Analytic<<std::endl;
    std::cout<<"Accuracy:       "<<(fabs(C-C_Analytic)/C_Analytic)*100<<" %"<<std::endl;
    return 0;

  }
