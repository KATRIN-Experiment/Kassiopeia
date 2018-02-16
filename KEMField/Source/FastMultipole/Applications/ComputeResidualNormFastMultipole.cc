#include <getopt.h>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <sys/stat.h>


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

#include "KMatrix.hh"
#include "KSquareMatrix.hh"
#include "KSimpleMatrix.hh"
#include "KSimpleSquareMatrix.hh"
#include "KVector.hh"
#include "KSimpleVector.hh"

#include "KElectrostaticBoundaryIntegratorFactory.hh"
#include "KBoundaryIntegralMatrix.hh"
#include "KBoundaryIntegralVector.hh"
#include "KBoundaryIntegralSolutionVector.hh"

#include "KGaussianElimination.hh"
#include "KRobinHood.hh"

#include "KEMFieldCanvas.hh"


#include "KIterativeKrylovSolver.hh"
#include "KPreconditionedIterativeKrylovSolver.hh"
#include "KIterativeKrylovRestartCondition.hh"
#include "KBiconjugateGradientStabilized.hh"
#include "KGeneralizedMinimalResidual.hh"
#include "KPreconditionedBiconjugateGradientStabilized.hh"
#include "KPreconditionedGeneralizedMinimalResidual.hh"

#include "KJacobiPreconditioner.hh"
#include "KBlockJacobiPreconditioner.hh"
#include "KImplicitKrylovPreconditioner.hh"

#include "KFMBoundaryIntegralMatrix.hh"
#include "KFMElectrostaticBoundaryIntegrator.hh"
#include "KFMElectrostaticFastMultipoleFieldSolver.hh"
#include "KFMElectrostaticBoundaryIntegratorEngine_SingleThread.hh"
#include "KFMElectrostaticFieldMapper_SingleThread.hh"
#include "KFMElectrostaticSparseBoundaryIntegralMatrix.hh"


#ifdef KEMFIELD_USE_OPENCL
#include "KFMElectrostaticFastMultipoleFieldSolver_OpenCL.hh"
#include "KFMElectrostaticFieldMapper_OpenCL.hh"
#include "KFMElectrostaticBoundaryIntegratorEngine_OpenCL.hh"
#include "KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL.hh"
#endif



#ifdef KEMFIELD_USE_VTK
#include "KEMVTKViewer.hh"
#include "KVTKIterationPlotter.hh"
#include "KEMVTKFieldCanvas.hh"
#endif

#include "KEMConstants.hh"

#include "KFMElectrostaticFastMultipoleFieldSolver.hh"
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
#endif

#ifndef DEFAULT_OUTPUT_DIR
#define DEFAULT_OUTPUT_DIR "."
#endif /* !DEFAULT_OUTPUT_DIR */

using namespace KGeoBag;
using namespace KEMField;

#ifdef KEMFIELD_USE_OPENCL
KFMElectrostaticFastMultipoleFieldSolver_OpenCL* fast_solver;
#else
KFMElectrostaticFastMultipoleFieldSolver* fast_solver;
#endif

#ifdef KEMFIELD_USE_OPENCL
KIntegratingFieldSolver<KOpenCLElectrostaticNumericBoundaryIntegrator>* direct_solver;
#else
KIntegratingFieldSolver<KElectrostaticBoundaryIntegrator>* direct_solver;
#endif



////////////////////////////////////////////////////////////////////////////////
//Configuration class for this test program

namespace KEMField{

class Configure_ComputeResidualNormFastMultipole: public KSAInputOutputObject
{
    public:

        Configure_ComputeResidualNormFastMultipole()
        {
            fVerbosity = 5;
            fDivisions = 3;
            fDegree = 3;
            fZeroMaskSize = 1;
            fMaxTreeLevel = 3;
            fUseRegionEstimation = 1;
            fRegionExpansionFactor = 1.1;
            fWorldCenterX = 0.0;
            fWorldCenterY = 0.0;
            fWorldCenterZ = 0.0;
            fWorldLength = 0.0;
            fNEvaluations = 0;
            fMode=0;
            fGeometryInputFileName = "";
            fSurfaceContainerName = "";
            fDataOutputFileName = "";
        }

        virtual ~Configure_ComputeResidualNormFastMultipole(){;};

        int GetVerbosity() const {return fVerbosity;};
        void SetVerbosity(const int& n){fVerbosity = n;};

        int GetDivisions() const {return fDivisions;};
        void SetDivisions(const int& d){fDivisions = d;};

        int GetDegree() const {return fDegree;};
        void SetDegree(const int& deg){fDegree = deg;};

        int GetZeroMaskSize() const {return fZeroMaskSize;};
        void SetZeroMaskSize(const int& z){fZeroMaskSize = z;};

        int GetMaxTreeLevel() const {return fMaxTreeLevel;};
        void SetMaxTreeLevel(const int& t){fMaxTreeLevel = t;};

        int GetUseRegionEstimation() const {return fUseRegionEstimation;};
        void SetUseRegionEstimation(const int& r){fUseRegionEstimation = r;};

        double GetRegionExpansionFactor() const {return fRegionExpansionFactor;};
        void SetRegionExpansionFactor(const double& d){fRegionExpansionFactor = d;};

        double GetWorldCenterX() const {return fWorldCenterX;};
        void SetWorldCenterX(const double& d){fWorldCenterX = d;};

        double GetWorldCenterY() const {return fWorldCenterY;};
        void SetWorldCenterY(const double& d){fWorldCenterY = d;};

        double GetWorldCenterZ() const {return fWorldCenterZ;};
        void SetWorldCenterZ(const double& d){fWorldCenterZ = d;};

        double GetWorldLength() const {return fWorldLength;};
        void SetWorldLength(const double& d){fWorldLength = d;};

        int GetNEvaluations() const {return fNEvaluations;};
        void SetNEvaluations(const int& t){fNEvaluations = t;};

        int GetMode() const {return fMode;};
        void SetMode(const int& t){fMode = t;};

        std::string GetGeometryInputFileName() const {return fGeometryInputFileName;};
        void SetGeometryInputFileName(const std::string& geo){fGeometryInputFileName = geo;};

        std::string GetSurfaceContainerName() const {return fSurfaceContainerName;};
        void SetSurfaceContainerName(const std::string& n){fSurfaceContainerName = n;};

        std::string GetDataOutputFileName() const {return fDataOutputFileName;};
        void SetDataOutputFileName(const std::string& outfile){fDataOutputFileName = outfile;};

        void DefineOutputNode(KSAOutputNode* node) const
        {
            AddKSAOutputFor(Configure_ComputeResidualNormFastMultipole,Verbosity,int);
            AddKSAOutputFor(Configure_ComputeResidualNormFastMultipole,Divisions,int);
            AddKSAOutputFor(Configure_ComputeResidualNormFastMultipole,Degree,int);
            AddKSAOutputFor(Configure_ComputeResidualNormFastMultipole,ZeroMaskSize,int);
            AddKSAOutputFor(Configure_ComputeResidualNormFastMultipole,MaxTreeLevel,int);

            AddKSAOutputFor(Configure_ComputeResidualNormFastMultipole,UseRegionEstimation,int);
            AddKSAOutputFor(Configure_ComputeResidualNormFastMultipole,RegionExpansionFactor,double);
            AddKSAOutputFor(Configure_ComputeResidualNormFastMultipole,WorldCenterX,double);
            AddKSAOutputFor(Configure_ComputeResidualNormFastMultipole,WorldCenterY,double);
            AddKSAOutputFor(Configure_ComputeResidualNormFastMultipole,WorldCenterZ,double);
            AddKSAOutputFor(Configure_ComputeResidualNormFastMultipole,WorldLength,double);

            AddKSAOutputFor(Configure_ComputeResidualNormFastMultipole,NEvaluations,int);
            AddKSAOutputFor(Configure_ComputeResidualNormFastMultipole,Mode,int);

            AddKSAOutputFor(Configure_ComputeResidualNormFastMultipole,GeometryInputFileName,std::string);
            AddKSAOutputFor(Configure_ComputeResidualNormFastMultipole,SurfaceContainerName,std::string);
            AddKSAOutputFor(Configure_ComputeResidualNormFastMultipole,DataOutputFileName,std::string);
        }

        void DefineInputNode(KSAInputNode* node)
        {
            AddKSAInputFor(Configure_ComputeResidualNormFastMultipole,Verbosity,int);
            AddKSAInputFor(Configure_ComputeResidualNormFastMultipole,Divisions,int);
            AddKSAInputFor(Configure_ComputeResidualNormFastMultipole,Degree,int);
            AddKSAInputFor(Configure_ComputeResidualNormFastMultipole,ZeroMaskSize,int);
            AddKSAInputFor(Configure_ComputeResidualNormFastMultipole,MaxTreeLevel,int);

            AddKSAInputFor(Configure_ComputeResidualNormFastMultipole,UseRegionEstimation,int);
            AddKSAInputFor(Configure_ComputeResidualNormFastMultipole,RegionExpansionFactor,double);
            AddKSAInputFor(Configure_ComputeResidualNormFastMultipole,WorldCenterX,double);
            AddKSAInputFor(Configure_ComputeResidualNormFastMultipole,WorldCenterY,double);
            AddKSAInputFor(Configure_ComputeResidualNormFastMultipole,WorldCenterZ,double);
            AddKSAInputFor(Configure_ComputeResidualNormFastMultipole,WorldLength,double);

            AddKSAInputFor(Configure_ComputeResidualNormFastMultipole,NEvaluations,int);
            AddKSAInputFor(Configure_ComputeResidualNormFastMultipole,Mode,int);

            AddKSAInputFor(Configure_ComputeResidualNormFastMultipole,GeometryInputFileName,std::string);
            AddKSAInputFor(Configure_ComputeResidualNormFastMultipole,SurfaceContainerName,std::string);
            AddKSAInputFor(Configure_ComputeResidualNormFastMultipole,DataOutputFileName,std::string);
        }

        virtual std::string ClassName() const {return std::string("Configure_ComputeResidualNormFastMultipole");};

    protected:

        int fVerbosity;
        int fDivisions;
        int fDegree;
        int fZeroMaskSize;
        int fMaxTreeLevel;
        int fUseRegionEstimation;
        double fRegionExpansionFactor;
        double fWorldCenterX;
        double fWorldCenterY;
        double fWorldCenterZ;
        double fWorldLength;
        int fNEvaluations;
        int fMode;
        std::string fGeometryInputFileName;
        std::string fSurfaceContainerName;
        std::string fDataOutputFileName;

};

DefineKSAClassName( Configure_ComputeResidualNormFastMultipole );

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

    KSAObjectInputNode< Configure_ComputeResidualNormFastMultipole >* config_input = new KSAObjectInputNode< Configure_ComputeResidualNormFastMultipole >(std::string("Configure_ComputeResidualNormFastMultipole"));

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

    int verbose = config_input->GetObject()->GetVerbosity();
    int divisions  = config_input->GetObject()->GetDivisions();
    int degree = config_input->GetObject()->GetDegree();
    int zeromask = config_input->GetObject()->GetZeroMaskSize();
    int max_tree_depth  = config_input->GetObject()->GetMaxTreeLevel();
    int use_region_estimation = config_input->GetObject()->GetUseRegionEstimation();
    double region_expansion_factor = config_input->GetObject()->GetRegionExpansionFactor();
    double worldx = config_input->GetObject()->GetWorldCenterX();
    double worldy = config_input->GetObject()->GetWorldCenterY();
    double worldz = config_input->GetObject()->GetWorldCenterZ();
    double length = config_input->GetObject()->GetWorldLength();

    unsigned int restart = 1000;
    double tolerance = 1e-4;

    //now we want to construct the tree
    KFMElectrostaticParameters params;
    params.divisions = divisions;
    params.degree = degree;
    params.zeromask = zeromask;
    params.maximum_tree_depth = max_tree_depth;
    params.region_expansion_factor = region_expansion_factor;
    params.use_region_estimation = use_region_estimation;
    params.world_center_x = worldx;
    params.world_center_y = worldy;
    params.world_center_z = worldz;
    params.world_length = length;
    params.use_caching = true;
    params.verbosity = verbose;

    unsigned int NEvaluations = config_input->GetObject()->GetNEvaluations();

    int mode = config_input->GetObject()->GetMode();

    std::string geometry_file_name = config_input->GetObject()->GetGeometryInputFileName();
    std::string container_name = config_input->GetObject()->GetSurfaceContainerName();
    std::string data_outfile = config_input->GetObject()->GetDataOutputFileName();

    if(verbose == 0)
    {
        KEMField::cout.Verbose(false);
    }
    else
    {
        KEMField::cout.Verbose(verbose);
    }


    //Read in the geometry file
    std::string suffix = geometry_file_name.substr(geometry_file_name.find_last_of("."),std::string::npos);
    KSurfaceContainer surfaceContainer;

    struct stat fileInfo;
    bool exists;
    int fileStat;

    // Attempt to get the file attributes
    fileStat = stat(geometry_file_name.c_str(),&fileInfo);
    if(fileStat == 0)
    exists = true;
    else
    exists = false;

    if (!exists)
    {
        std::cout<<"Error: file \""<<geometry_file_name<<"\" cannot be read."<<std::endl;
        return 1;
    }

    KBinaryDataStreamer binaryDataStreamer;

    if (suffix.compare(binaryDataStreamer.GetFileSuffix()) != 0)
    {
        std::cout<<"Error: unkown file extension \""<<suffix<<"\""<<std::endl;
        return 1;
    }

    KEMFileInterface::GetInstance()->Read(geometry_file_name,surfaceContainer,container_name);

    ////////////////////////////////////////////////////////////////////////////
    //define integrator types
    #ifdef KEMFIELD_USE_OPENCL
        typedef KFMElectrostaticBoundaryIntegrator<KFMElectrostaticBoundaryIntegratorEngine_OpenCL> FastMultipoleEBI;
    #else
        typedef KFMElectrostaticBoundaryIntegrator<KFMElectrostaticBoundaryIntegratorEngine_SingleThread> FastMultipoleEBI;
    #endif

    //now we create a fast multipole BEM matrix
    FastMultipoleEBI* fm_integrator = new FastMultipoleEBI(surfaceContainer);

    fm_integrator->SetUniqueIDString("residual_norm");
    fm_integrator->Initialize(params);

    KFMBoundaryIntegralMatrix< FastMultipoleEBI > fmA(surfaceContainer, *fm_integrator);
    KBoundaryIntegralSolutionVector< FastMultipoleEBI > fmx(surfaceContainer, *fm_integrator);
    KBoundaryIntegralVector< FastMultipoleEBI > fmb(surfaceContainer, *fm_integrator);

    unsigned int n_elem = surfaceContainer.size();
    KSimpleVector<FastMultipoleEBI::ValueType> b_prime;
    b_prime.resize(n_elem, 0.);

    //now use the fast multipole matrix to calculate A*x = b'
    fmA.Multiply(fmx, b_prime);

////////////////////////////////////////////////////////////////////////////////

    //First lets compute the l2 norm of b' and b
    double inf_norm = 0;
    double l2_norm = 0.;
    double b_mag = 0.;
    double b_max = 0;
    for(unsigned int i=0; i<n_elem; i++)
    {
        if( std::fabs(fmb(i)) > b_max){b_max = std::fabs( fmb(i) );};
        double del = fmb(i) - b_prime(i);
        if( std::fabs(del) > inf_norm){inf_norm = std::fabs(del);};
        l2_norm += del*del;
        b_mag += fmb(i)*fmb(i);
    }

    l2_norm = std::sqrt(l2_norm);
    b_mag = std::sqrt(b_mag);

    std::cout<<"absolute l2_norm w/ original B.C. |b-b'| = "<<l2_norm<<std::endl;
    std::cout<<"relative l2 norm w/ original B.C. |b-b'|/|b| = "<<l2_norm/b_mag<<std::endl;
    std::cout<<"absolute infinity norm w.r.t. original B.C. = "<<inf_norm<<std::endl;
    std::cout<<"relative infinity norm w.r.t. original B.C. = "<<inf_norm/b_max<<std::endl;

    return 0;
  }
