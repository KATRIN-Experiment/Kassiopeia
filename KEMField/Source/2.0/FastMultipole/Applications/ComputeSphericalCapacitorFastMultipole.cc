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


#include "KSADataStreamer.hh"
#include "KBinaryDataStreamer.hh"
#include "KSerializer.hh"
#include "KSADataStreamer.hh"
#include "KBinaryDataStreamer.hh"
#include "KEMFileInterface.hh"

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

#include "KIterativeKrylovSolver.hh"
#include "KPreconditionedIterativeKrylovSolver.hh"
#include "KIterativeKrylovRestartCondition.hh"
#include "KBiconjugateGradientStabilized.hh"
#include "KGeneralizedMinimalResidual.hh"
#include "KPreconditionedBiconjugateGradientStabilized.hh"
#include "KPreconditionedGeneralizedMinimalResidual.hh"
#include "KRobinHoodPreconditioner.hh"
#include "KJacobiPreconditioner.hh"
#include "KIdentityPreconditioner.hh"


#include "KFMBoundaryIntegralMatrix.hh"
#include "KFMElectrostaticBoundaryIntegrator.hh"
#include "KFMElectrostaticFastMultipoleFieldSolver.hh"
#include "KFMElectrostaticBoundaryIntegratorEngine_SingleThread.hh"
#include "KFMElectrostaticFieldMapper_SingleThread.hh"

#include "KElectrostaticIntegratingFieldSolver.hh"

#include "KFMElectrostaticTree.hh"
#include "KFMElectrostaticTreeConstructor.hh"
#include "KFMElectrostaticFastMultipoleFieldSolver.hh"

#include "KFMNamedScalarData.hh"
#include "KFMNamedScalarDataCollection.hh"


#ifdef KEMFIELD_USE_OPENCL
#include "KFMElectrostaticFastMultipoleFieldSolver_OpenCL.hh"
#include "KFMElectrostaticFieldMapper_OpenCL.hh"
#include "KFMElectrostaticBoundaryIntegratorEngine_OpenCL.hh"
#include "KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL.hh"
#include "KOpenCLSurfaceContainer.hh"
#include "KOpenCLElectrostaticBoundaryIntegrator.hh"
#include "KOpenCLBoundaryIntegralMatrix.hh"
#include "KOpenCLBoundaryIntegralVector.hh"
#include "KOpenCLBoundaryIntegralSolutionVector.hh"
#include "KRobinHood_OpenCL.hh"
#include "KOpenCLElectrostaticIntegratingFieldSolver.hh"
#endif
#ifndef DEFAULT_OUTPUT_DIR
#define DEFAULT_OUTPUT_DIR "."
#endif /* !DEFAULT_OUTPUT_DIR */

using namespace KGeoBag;
using namespace KEMField;


////////////////////////////////////////////////////////////////////////////////
//Configuration class for this test program

namespace KEMField{

class Configure_ComputeSphericalCapacitorFastMultipole: public KSAInputOutputObject
{
    public:

        Configure_ComputeSphericalCapacitorFastMultipole()
        {
            fVerbosity = 5;
            fScale = 1.0;
            fAccuracy = 1e-4;
            fDivisions = 3;
            fDegree = 3;
            fZeroMaskSize = 1;
            fMaxTreeLevel = 3;
            fKrylovMethod = 3;
            fUsePlot = 0;
            fGMRESRestartParameter = 30;
            fSaveOutput = 0;
            fOutputFileName = "";
        }

        virtual ~Configure_ComputeSphericalCapacitorFastMultipole(){;};


        int GetVerbosity() const {return fVerbosity;};
        void SetVerbosity(const int& n){fVerbosity = n;};

        double GetAccuracy() const {return fAccuracy;};
        void SetAccuracy(const double& n){fAccuracy = n;};

        int GetScale() const {return fScale;};
        void SetScale(const int& s){fScale = s;};

        int GetDivisions() const {return fDivisions;};
        void SetDivisions(const int& d){fDivisions = d;};

        int GetDegree() const {return fDegree;};
        void SetDegree(const int& deg){fDegree = deg;};

        int GetZeroMaskSize() const {return fZeroMaskSize;};
        void SetZeroMaskSize(const int& z){fZeroMaskSize = z;};

        int GetMaxTreeLevel() const {return fMaxTreeLevel;};
        void SetMaxTreeLevel(const int& t){fMaxTreeLevel = t;};

        int GetKrylovMethod() const {return fKrylovMethod;};
        void SetKrylovMethod(const int& m){fKrylovMethod = m;};

        int GetGMRESRestartParameter() const {return fGMRESRestartParameter;};
        void SetGMRESRestartParameter( const int& r){fGMRESRestartParameter = r;};

        int GetUsePlot() const {return fUsePlot;};
        void SetUsePlot(const int& b){fUsePlot = b;};

        int GetSaveOutput() const {return fSaveOutput;};
        void SetSaveOutput(const int& b){fSaveOutput = b;};

        std::string GetOutputFileName() const {return fOutputFileName;};
        void SetOutputFileName(const std::string& outfile){fOutputFileName = outfile;};

        void DefineOutputNode(KSAOutputNode* node) const
        {
            AddKSAOutputFor(Configure_ComputeSphericalCapacitorFastMultipole,Verbosity,int);
            AddKSAOutputFor(Configure_ComputeSphericalCapacitorFastMultipole,Accuracy,double);
            AddKSAOutputFor(Configure_ComputeSphericalCapacitorFastMultipole,Scale,int);
            AddKSAOutputFor(Configure_ComputeSphericalCapacitorFastMultipole,Divisions,int);
            AddKSAOutputFor(Configure_ComputeSphericalCapacitorFastMultipole,Degree,int);
            AddKSAOutputFor(Configure_ComputeSphericalCapacitorFastMultipole,ZeroMaskSize,int);
            AddKSAOutputFor(Configure_ComputeSphericalCapacitorFastMultipole,MaxTreeLevel,int);
            AddKSAOutputFor(Configure_ComputeSphericalCapacitorFastMultipole,KrylovMethod,int);
            AddKSAOutputFor(Configure_ComputeSphericalCapacitorFastMultipole,UsePlot,int);
            AddKSAOutputFor(Configure_ComputeSphericalCapacitorFastMultipole,GMRESRestartParameter,int);
            AddKSAOutputFor(Configure_ComputeSphericalCapacitorFastMultipole,SaveOutput,int);
            AddKSAOutputFor(Configure_ComputeSphericalCapacitorFastMultipole,OutputFileName,std::string);
        }

        void DefineInputNode(KSAInputNode* node)
        {
            AddKSAInputFor(Configure_ComputeSphericalCapacitorFastMultipole,Verbosity,int);
            AddKSAInputFor(Configure_ComputeSphericalCapacitorFastMultipole,Accuracy,double);
            AddKSAInputFor(Configure_ComputeSphericalCapacitorFastMultipole,Scale,int);
            AddKSAInputFor(Configure_ComputeSphericalCapacitorFastMultipole,Divisions,int);
            AddKSAInputFor(Configure_ComputeSphericalCapacitorFastMultipole,Degree,int);
            AddKSAInputFor(Configure_ComputeSphericalCapacitorFastMultipole,ZeroMaskSize,int);
            AddKSAInputFor(Configure_ComputeSphericalCapacitorFastMultipole,MaxTreeLevel,int);
            AddKSAInputFor(Configure_ComputeSphericalCapacitorFastMultipole,KrylovMethod,int);
            AddKSAInputFor(Configure_ComputeSphericalCapacitorFastMultipole,UsePlot,int);
            AddKSAInputFor(Configure_ComputeSphericalCapacitorFastMultipole,GMRESRestartParameter,int);
            AddKSAInputFor(Configure_ComputeSphericalCapacitorFastMultipole,SaveOutput,int);
            AddKSAInputFor(Configure_ComputeSphericalCapacitorFastMultipole,OutputFileName,std::string);
        }

        virtual std::string ClassName() const {return std::string("Configure_ComputeSphericalCapacitorFastMultipole");};

    protected:

        double fAccuracy;
        int fScale;
        int fVerbosity;
        int fDivisions;
        int fDegree;
        int fZeroMaskSize;
        int fMaxTreeLevel;
        int fKrylovMethod;
        int fUsePlot;
        int fGMRESRestartParameter;
        int fSaveOutput;
        std::string fOutputFileName;

};

DefineKSAClassName( Configure_ComputeSphericalCapacitorFastMultipole );

}


////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{

    if(argc < 2)
    {
        kfmout<<"Please specify the full path to the configuration file."<<kfmendl;
        return 1;
    }

    std::string input_file(argv[1]);

    KSAFileReader reader;
    reader.SetFileName(input_file);

    KSAInputCollector* in_collector = new KSAInputCollector();
    in_collector->SetFileReader(&reader);

    KSAObjectInputNode< Configure_ComputeSphericalCapacitorFastMultipole >* config_input = new KSAObjectInputNode< Configure_ComputeSphericalCapacitorFastMultipole >(std::string("Configure_ComputeSphericalCapacitorFastMultipole"));

    kfmout<<"Reading configuration file. "<<kfmendl;

    if( reader.Open() )
    {
        in_collector->ForwardInput(config_input);
    }
    else
    {
        kfmout<<"Could not open configuration file."<<kfmendl;
        return 1;
    }

    double accuracy = config_input->GetObject()->GetAccuracy();
    int scale = config_input->GetObject()->GetScale();
    int verbose = config_input->GetObject()->GetVerbosity();
    int divisions = config_input->GetObject()->GetDivisions();
    int degree = config_input->GetObject()->GetDegree();
    int zeromask = config_input->GetObject()->GetZeroMaskSize();
    int max_tree_depth = config_input->GetObject()->GetMaxTreeLevel();
    int method = config_input->GetObject()->GetKrylovMethod();
    int gmres_restart = config_input->GetObject()->GetGMRESRestartParameter();

    bool usePlot = false;
    int plot = config_input->GetObject()->GetUsePlot();
    if(plot != 0){usePlot = true;};

    bool saveOutput = false;
    int save_status = config_input->GetObject()->GetSaveOutput();
    if(save_status != 0){saveOutput = true;};

    std::string outfile = config_input->GetObject()->GetOutputFileName();

    if(verbose == 0)
    {
        KEMField::cout.Verbose(false);
    }
    else
    {
        KEMField::cout.Verbose(verbose);
    }

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
    params.verbosity = verbose;
    params.use_caching = true;

#ifndef KEMFIELD_USE_OPENCL

    typedef KFMElectrostaticBoundaryIntegrator<KFMElectrostaticBoundaryIntegratorEngine_SingleThread> KFMSingleThreadEBI;
    KFMSingleThreadEBI integrator(surfaceContainer);
    integrator.Initialize(params);
    KFMBoundaryIntegralMatrix< KFMSingleThreadEBI > A(surfaceContainer,integrator);
    KBoundaryIntegralSolutionVector< KFMSingleThreadEBI > x(surfaceContainer,integrator);
    KBoundaryIntegralVector< KFMSingleThreadEBI > b(surfaceContainer,integrator);

    if(method == 1)
    {
        KIterativeKrylovSolver< KFMSingleThreadEBI::ValueType, KBiconjugateGradientStabilized> biCGSTAB;
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

        KJacobiPreconditioner<KFMSingleThreadEBI::ValueType> P(A);

        KPreconditionedIterativeKrylovSolver< KFMSingleThreadEBI::ValueType, KPreconditionedBiconjugateGradientStabilized> pbiCGSTAB;
        pbiCGSTAB.SetTolerance(accuracy);

        pbiCGSTAB.AddVisitor(new KIterationDisplay<double>());

    #ifdef KEMFIELD_USE_VTK
        if(usePlot)
        {
            pbiCGSTAB.AddVisitor(new KVTKIterationPlotter<double>());
        }
    #endif

        pbiCGSTAB.Solve(A,P,x,b);
    }
    else if(method == 3)
    {
        KIterativeKrylovRestartCondition* restart_cond = new KIterativeKrylovRestartCondition();
        restart_cond->SetNumberOfIterationsBetweenRestart(gmres_restart);


        KIterativeKrylovSolver< KFMSingleThreadEBI::ValueType, KGeneralizedMinimalResidual> gmres;
        gmres.SetTolerance(accuracy);
        gmres.SetRestartCondition(restart_cond);

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


    typedef KFMElectrostaticBoundaryIntegrator<KFMElectrostaticBoundaryIntegratorEngine_OpenCL> KFMOpenCLEBI;
    KFMOpenCLEBI integrator(surfaceContainer);
    integrator.Initialize(params);
    KFMBoundaryIntegralMatrix< KFMOpenCLEBI > A(surfaceContainer,integrator);
    KBoundaryIntegralSolutionVector<KFMOpenCLEBI > x(surfaceContainer,integrator);
    KBoundaryIntegralVector< KFMOpenCLEBI> b(surfaceContainer,integrator);

    if(method == 1)
    {

        KIterativeKrylovSolver< KFMOpenCLEBI::ValueType, KBiconjugateGradientStabilized> biCGSTAB;
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

        KJacobiPreconditioner<KFMOpenCLEBI::ValueType> P(A);

        KPreconditionedIterativeKrylovSolver< KFMOpenCLEBI::ValueType, KPreconditionedBiconjugateGradientStabilized> biCGSTAB;
        biCGSTAB.SetTolerance(accuracy);


        biCGSTAB.AddVisitor(new KIterationDisplay<double>());

    #ifdef KEMFIELD_USE_VTK
        if(usePlot)
        {
            biCGSTAB.AddVisitor(new KVTKIterationPlotter<double>());
        }
    #endif

        biCGSTAB.Solve(A,P,x,b);
    }
    else if(method == 3)
    {
        KIterativeKrylovRestartCondition* restart_cond = new KIterativeKrylovRestartCondition();
        restart_cond->SetNumberOfIterationsBetweenRestart(gmres_restart);

        KIterativeKrylovSolver< KFMOpenCLEBI::ValueType, KGeneralizedMinimalResidual> gmres;
        gmres.SetTolerance(accuracy);
        gmres.SetRestartCondition(restart_cond);

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

    std::cout<<std::setprecision(14)<<"Capacitance:    "<<C<<std::endl;
    std::cout.setf( std::ios::fixed, std:: ios::floatfield );
    std::cout<<std::setprecision(14)<<"Accepted value: "<<C_Analytic<<std::endl;
    std::cout<<"Accuracy:       "<<(fabs(C-C_Analytic)/C_Analytic)*100<<" %"<<std::endl;



    if(saveOutput)
    {
        std::cout<<"saving surface container data"<<std::endl;

//        KMetadataStreamer mDS;
//        mDS.open(outfile + std::string(".smd"),"overwrite");
//        mDS << surfaceContainer;
//        mDS.close();

//        KBinaryDataStreamer bDS;
//        bDS.open(outfile + std::string(".kbd"),"overwrite");
//        bDS << surfaceContainer;
//        bDS.close();

//        KSADataStreamer saS;
//        saS.open(outfile + std::string(".ksa"),"overwrite");
//        saS << surfaceContainer;
//        saS.close();

        KEMFileInterface::GetInstance()->Write(surfaceContainer,"surfaceContainer");
    }


    return 0;

  }
