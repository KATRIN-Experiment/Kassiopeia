#include <getopt.h>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <sys/stat.h>

#include "KElectrostaticBoundaryIntegratorFactory.hh"
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

#include "KFMElectrostaticFastMultipoleFieldSolver.hh"
#include "KFMElectrostaticFieldMapper_SingleThread.hh"
#include "KElectrostaticIntegratingFieldSolver.hh"

#include "KFMElectrostaticTree.hh"
#include "KFMElectrostaticTreeConstructor.hh"
#include "KFMElectrostaticFastMultipoleFieldSolver.hh"
#include "KFMElementAspectRatioExtractor.hh"

#include "KFMNamedScalarData.hh"
#include "KFMNamedScalarDataCollection.hh"


#ifdef KEMFIELD_USE_OPENCL
#include "KOpenCLElectrostaticBoundaryIntegratorFactory.hh"
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
KIntegratingFieldSolver<KOpenCLElectrostaticBoundaryIntegrator>* direct_solver;
#else
KIntegratingFieldSolver<KElectrostaticBoundaryIntegrator>* direct_solver;
#endif



////////////////////////////////////////////////////////////////////////////////
//Configuration class for this test program

namespace KEMField{

class Configure_AccuracyComparisonFastMultipole: public KSAInputOutputObject
{
    public:

        Configure_AccuracyComparisonFastMultipole()
        {
            fVerbosity = 5;
            fTopLevelDivisions = 3;
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
            fMode=6;
            fAllowedElementMode = 0;
            fGeometryInputFileName = "";
            fSurfaceContainerName = "";
            fDataOutputFileName = "";
        }

        virtual ~Configure_AccuracyComparisonFastMultipole(){;};

        int GetVerbosity() const {return fVerbosity;};
        void SetVerbosity(const int& n){fVerbosity = n;};

        int GetTopLevelDivisions() const {return fTopLevelDivisions;};
        void SetTopLevelDivisions(const int& d){fTopLevelDivisions = d;};

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

        int GetAllowedElementMode() const {return fAllowedElementMode;};
        void SetAllowedElementMode(const int& t){fAllowedElementMode = t;};

        std::string GetGeometryInputFileName() const {return fGeometryInputFileName;};
        void SetGeometryInputFileName(const std::string& geo){fGeometryInputFileName = geo;};

        std::string GetSurfaceContainerName() const {return fSurfaceContainerName;};
        void SetSurfaceContainerName(const std::string& n){fSurfaceContainerName = n;};

        std::string GetDataOutputFileName() const {return fDataOutputFileName;};
        void SetDataOutputFileName(const std::string& outfile){fDataOutputFileName = outfile;};

        void DefineOutputNode(KSAOutputNode* node) const
        {
            AddKSAOutputFor(Configure_AccuracyComparisonFastMultipole,Verbosity,int);
            AddKSAOutputFor(Configure_AccuracyComparisonFastMultipole,TopLevelDivisions,int);
            AddKSAOutputFor(Configure_AccuracyComparisonFastMultipole,Divisions,int);
            AddKSAOutputFor(Configure_AccuracyComparisonFastMultipole,Degree,int);
            AddKSAOutputFor(Configure_AccuracyComparisonFastMultipole,ZeroMaskSize,int);
            AddKSAOutputFor(Configure_AccuracyComparisonFastMultipole,MaxTreeLevel,int);

            AddKSAOutputFor(Configure_AccuracyComparisonFastMultipole,UseRegionEstimation,int);
            AddKSAOutputFor(Configure_AccuracyComparisonFastMultipole,RegionExpansionFactor,double);
            AddKSAOutputFor(Configure_AccuracyComparisonFastMultipole,WorldCenterX,double);
            AddKSAOutputFor(Configure_AccuracyComparisonFastMultipole,WorldCenterY,double);
            AddKSAOutputFor(Configure_AccuracyComparisonFastMultipole,WorldCenterZ,double);
            AddKSAOutputFor(Configure_AccuracyComparisonFastMultipole,WorldLength,double);

            AddKSAOutputFor(Configure_AccuracyComparisonFastMultipole,NEvaluations,int);
            AddKSAOutputFor(Configure_AccuracyComparisonFastMultipole,Mode,int);
            AddKSAOutputFor(Configure_AccuracyComparisonFastMultipole,AllowedElementMode,int);

            AddKSAOutputFor(Configure_AccuracyComparisonFastMultipole,GeometryInputFileName,std::string);
            AddKSAOutputFor(Configure_AccuracyComparisonFastMultipole,SurfaceContainerName,std::string);
            AddKSAOutputFor(Configure_AccuracyComparisonFastMultipole,DataOutputFileName,std::string);
        }

        void DefineInputNode(KSAInputNode* node)
        {
            AddKSAInputFor(Configure_AccuracyComparisonFastMultipole,Verbosity,int);
            AddKSAInputFor(Configure_AccuracyComparisonFastMultipole,TopLevelDivisions,int);
            AddKSAInputFor(Configure_AccuracyComparisonFastMultipole,Divisions,int);
            AddKSAInputFor(Configure_AccuracyComparisonFastMultipole,Degree,int);
            AddKSAInputFor(Configure_AccuracyComparisonFastMultipole,ZeroMaskSize,int);
            AddKSAInputFor(Configure_AccuracyComparisonFastMultipole,MaxTreeLevel,int);

            AddKSAInputFor(Configure_AccuracyComparisonFastMultipole,UseRegionEstimation,int);
            AddKSAInputFor(Configure_AccuracyComparisonFastMultipole,RegionExpansionFactor,double);
            AddKSAInputFor(Configure_AccuracyComparisonFastMultipole,WorldCenterX,double);
            AddKSAInputFor(Configure_AccuracyComparisonFastMultipole,WorldCenterY,double);
            AddKSAInputFor(Configure_AccuracyComparisonFastMultipole,WorldCenterZ,double);
            AddKSAInputFor(Configure_AccuracyComparisonFastMultipole,WorldLength,double);

            AddKSAInputFor(Configure_AccuracyComparisonFastMultipole,NEvaluations,int);
            AddKSAInputFor(Configure_AccuracyComparisonFastMultipole,Mode,int);
            AddKSAInputFor(Configure_AccuracyComparisonFastMultipole,AllowedElementMode,int);

            AddKSAInputFor(Configure_AccuracyComparisonFastMultipole,GeometryInputFileName,std::string);
            AddKSAInputFor(Configure_AccuracyComparisonFastMultipole,SurfaceContainerName,std::string);
            AddKSAInputFor(Configure_AccuracyComparisonFastMultipole,DataOutputFileName,std::string);
        }

        virtual std::string ClassName() const {return std::string("Configure_AccuracyComparisonFastMultipole");};

    protected:

        int fVerbosity;
        int fTopLevelDivisions;
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
        int fAllowedElementMode; //0=all, 1=triangles only, 2=rectangles only, 3=wires only
        std::string fGeometryInputFileName;
        std::string fSurfaceContainerName;
        std::string fDataOutputFileName;

};

DefineKSAClassName( Configure_AccuracyComparisonFastMultipole );

class SelectBoundaryElements:
public KSelectiveVisitor<KShapeVisitor,KTYPELIST_3(KTriangle, KRectangle, KLineSegment)>
{
    public:

        SelectBoundaryElements():fAllowedElementMode(0),fMaxAspectRatio(100),fTarget(NULL),fSource(NULL){};
        ~SelectBoundaryElements(){;};

        void SetAllowedElementMode(int mode){fAllowedElementMode = mode;};
        void SetMaximumAspectRatio(double ar){fMaxAspectRatio = ar;};
        void SetID(unsigned int i){fID = i;};

        void SetTargetSurfaceContainer(KSurfaceContainer* c)
        {
            fTarget = c;
        }

        void SetSourceSurfaceContainer(KSurfaceContainer* c)
        {
            fSource = c;
        }

        void Visit(KTriangle& /*t*/)
        {
            if(fAllowedElementMode == 0 || fAllowedElementMode == 1)
            {
                fSource->at(fID)->Accept(fAspectRatioExtractor);
                double ar = fAspectRatioExtractor.GetAspectRatio();

                if(ar < fMaxAspectRatio)
                {
                    fTarget->push_back( fSource->at(fID) );
                }
            }
        };

        void Visit(KRectangle& /*r*/)
        {
            if(fAllowedElementMode == 0 || fAllowedElementMode == 2)
            {
                fSource->at(fID)->Accept(fAspectRatioExtractor);
                double ar = fAspectRatioExtractor.GetAspectRatio();

                if(ar < fMaxAspectRatio)
                {
                    fTarget->push_back( fSource->at(fID) );
                }
            }
        };

        void Visit(KLineSegment& /*l*/)
        {
            if(fAllowedElementMode == 0 || fAllowedElementMode == 3)
            {
                //no aspect ratio test for wires currently
                fTarget->push_back( fSource->at(fID) );
            }
        };

    private:

        int fAllowedElementMode;
        unsigned int fID;
        double fMaxAspectRatio;
        KSurfaceContainer* fTarget;
        KSurfaceContainer* fSource;

        KFMElementAspectRatioExtractor fAspectRatioExtractor;

};

} //end of kemfield namespace

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

    KSAObjectInputNode< Configure_AccuracyComparisonFastMultipole >* config_input = new KSAObjectInputNode< Configure_AccuracyComparisonFastMultipole >(std::string("Configure_AccuracyComparisonFastMultipole"));

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
    int top_level_divisions  = config_input->GetObject()->GetTopLevelDivisions();
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


    //now we want to construct the tree
    KFMElectrostaticParameters params;
    params.top_level_divisions = top_level_divisions;
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

    //mode = 0 (random sample points)
    //mode = 1 (single line along x at y=0, z=0)
    //mode = 2 (single line along y at x=0, z=0)
    //mode = 3 (single line along z at x=0, y=0)
    //mode = 4 (two dimensional slice at x=0)
    //mode = 5 (two dimensional slice at  y=0)
    //mode = 6 (two dimensional slice at  z=0)
    //mode = 7 (three dimensional volume)
    int mode = config_input->GetObject()->GetMode();

    int allowed_element_mode = config_input->GetObject()->GetAllowedElementMode();

    std::cout<<"mode = "<<mode<<std::endl;

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

    KSurfaceContainer surfaceContainer;
    KSurfaceContainer* proxy_surfaceContainer = new KSurfaceContainer();

    KEMFileInterface::GetInstance()->Read(geometry_file_name, *proxy_surfaceContainer,container_name);

    //strip out certain geometric elements based on shape
    //TODO...allow ability to strip out elements based on area/aspect ratio
    SelectBoundaryElements bem_selector;
    bem_selector.SetAllowedElementMode( allowed_element_mode );
    bem_selector.SetSourceSurfaceContainer( proxy_surfaceContainer );
    bem_selector.SetTargetSurfaceContainer( &surfaceContainer );
    for(unsigned int i=0; i<proxy_surfaceContainer->size(); i++)
    {
        bem_selector.SetID(i);
        proxy_surfaceContainer->at(i)->Accept(bem_selector);
    }


    //now create the direct solver
    #ifdef KEMFIELD_USE_OPENCL
    KOpenCLSurfaceContainer* oclContainer;
    oclContainer = new KOpenCLSurfaceContainer( surfaceContainer );
    KOpenCLInterface::GetInstance()->SetActiveData( oclContainer );
    //KOpenCLElectrostaticNumericBoundaryIntegrator integrator(*oclContainer);
    KOpenCLElectrostaticBoundaryIntegrator integrator{KoclEBIFactory::MakeDefault(*oclContainer)};
    direct_solver = new KIntegratingFieldSolver<KOpenCLElectrostaticBoundaryIntegrator>(*oclContainer,integrator);
    direct_solver->Initialize();
    #else
    KElectrostaticBoundaryIntegrator integrator {KEBIFactory::MakeDefault()};
    direct_solver = new KIntegratingFieldSolver<KElectrostaticBoundaryIntegrator>(surfaceContainer,integrator);
    #endif


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

    //the tree constuctor definitions
    #ifdef KEMFIELD_USE_OPENCL
    typedef KFMElectrostaticTreeConstructor< KFMElectrostaticFieldMapper_OpenCL > TreeConstructor;
    #else
    typedef KFMElectrostaticTreeConstructor< KFMElectrostaticFieldMapper_SingleThread > TreeConstructor;
    #endif

    // compute hash of the solved geometry
    KMD5HashGenerator solutionHashGenerator;
    string solutionHash = solutionHashGenerator.GenerateHash( surfaceContainer );

    // compute hash of the parameter values on the bare geometry
    // compute hash of the parameter values
    KMD5HashGenerator parameterHashGenerator;
    string parameterHash = parameterHashGenerator.GenerateHash( params );

    // create label set for multipole tree container object
    string fmContainerBase( KFMElectrostaticTreeData::Name() );
    string fmContainerName = fmContainerBase + string( "_" ) + solutionHash + string( "_" ) + parameterHash;
    vector< string > fmContainerLabels;
    fmContainerLabels.push_back( fmContainerBase );
    fmContainerLabels.push_back( solutionHash );
    fmContainerLabels.push_back( parameterHash );

    KFMElectrostaticTreeData* tree_data = new KFMElectrostaticTreeData();
    KFMElectrostaticTree* tree = new KFMElectrostaticTree();

    bool containerFound = false;
    KEMFileInterface::GetInstance()->FindByLabels( *tree_data, fmContainerLabels, 0, containerFound);

    if( containerFound == true )
    {
        std::cout << "fast multipole tree found." << std::endl;

        //construct tree from data
        TreeConstructor constructor;
        constructor.ConstructTree(*tree_data, *tree);
    }
    else
    {
        std::cout<< "no fast multipole tree found." << std::endl;

        //must construct the tree
        //assign tree parameters and id
        tree->SetParameters(params);
        tree->GetTreeProperties()->SetTreeID(fmContainerName);

        //construct the tree
        TreeConstructor constructor;
        #ifdef KEMFIELD_USE_OPENCL
        constructor.ConstructTree(*oclContainer, *tree);
        #else
        constructor.ConstructTree(surfaceContainer, *tree);
        #endif

        constructor.SaveTree(*tree, *tree_data);

        KEMFileInterface::GetInstance()->Write( *tree_data, fmContainerName, fmContainerLabels );
    }

    //now build the field solver
    #ifdef KEMFIELD_USE_OPENCL
        fast_solver = new KFMElectrostaticFastMultipoleFieldSolver_OpenCL(KoclEBIFactory::MakeDefaultConfig(),*oclContainer, *tree);
    #else
        fast_solver = new KFMElectrostaticFastMultipoleFieldSolver(surfaceContainer, *tree);
    #endif



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

    KFMNamedScalarData tree_level; tree_level.SetName("tree_level");
    KFMNamedScalarData n_direct_calls; n_direct_calls.SetName("n_direct_calls");

    KFMNamedScalarData fmm_time_per_potential_call; fmm_time_per_potential_call.SetName("fmm_time_per_potential_call");
    KFMNamedScalarData fmm_time_per_field_call; fmm_time_per_field_call.SetName("fmm_time_per_field_call");
    KFMNamedScalarData direct_time_per_potential_call; direct_time_per_potential_call.SetName("direct_time_per_potential_call");
    KFMNamedScalarData direct_time_per_field_call; direct_time_per_field_call.SetName("direct_time_per_field_call");

    //compute the positions of the evaluation points
    KFMCube<3>* world = KFMObjectRetriever<KFMElectrostaticNodeObjects, KFMCube<3> >::GetNodeObject(tree->GetRootNode());
    double world_length = world->GetLength();
    double length_a = world_length/2.0 - 0.001*world_length;
    double length_b = world_length/2.0 - 0.001*world_length;
    double length_c = world_length/2.0 - 0.001*world_length;
    KFMPoint<3> center = world->GetCenter();

    KEMThreeVector direction_a;
    KEMThreeVector direction_b;
    KEMThreeVector direction_c;

    KEMThreeVector p0(center[0], center[1], center[2]);
    KEMThreeVector point;

    unsigned int n_points = 0;
    KEMThreeVector* points = NULL;

    switch( mode )
    {
        case 0:
            n_points = NEvaluations;
            points = new KEMThreeVector[n_points];
            direction_a[0] = 1.0; direction_a[1] = 0.0; direction_a[2] = 0.0;
            direction_b[0] = 0.0; direction_b[1] = 1.0; direction_b[2] = 0.0;
            direction_c[0] = 0.0; direction_c[1] = 0.0; direction_c[2] = 1.0;
            p0 = p0 - length_a*direction_a;
            p0 = p0 - length_b*direction_b;
            p0 = p0 - length_c*direction_c;
            for(unsigned int i=0; i<NEvaluations; i++)
            {
                double rand_x = ((double)rand()/RAND_MAX);
                double rand_y = ((double)rand()/RAND_MAX);
                double rand_z = ((double)rand()/RAND_MAX);

                //std::cout<<"rand = "<<rand_x<<", "<<rand_y<<", "<<rand_z<<std::endl;

                double dist_a = rand_x*2.0*length_a;
                double dist_b = rand_y*2.0*length_b;
                double dist_c = rand_z*2.0*length_c;
                point = p0 + dist_a*direction_a + dist_b*direction_b + dist_c*direction_c;

                x_coord.AddNextValue(point[0]);
                y_coord.AddNextValue(point[1]);
                z_coord.AddNextValue(point[2]);

                points[i] = point;
            }
        break;
        case 1:
            n_points = NEvaluations;
            points = new KEMThreeVector[n_points];
            direction_a[0] = 1.0; direction_a[1] = 0.0; direction_a[2] = 0.0;
            p0 = p0 - length_a*direction_a;
            for(unsigned int i=0; i<NEvaluations; i++)
            {
                point = p0 + i*(2.0*length_a/NEvaluations)*direction_a;
                x_coord.AddNextValue(point[0]);
                y_coord.AddNextValue(point[1]);
                z_coord.AddNextValue(point[2]);
                points[i] = point;
            }
        break;
        case 2:
            n_points = NEvaluations;
            points = new KEMThreeVector[n_points];
            direction_a[0] = 0.0; direction_a[1] = 1.0; direction_a[2] = 0.0;
            p0 = p0 - length_a*direction_a;
            for(unsigned int i=0; i<NEvaluations; i++)
            {
                point = p0 + i*(2.0*length_a/NEvaluations)*direction_a;
                x_coord.AddNextValue(point[0]);
                y_coord.AddNextValue(point[1]);
                z_coord.AddNextValue(point[2]);
                points[i] = point;
            }
        break;
        case 3:
            n_points = NEvaluations;
            points = new KEMThreeVector[n_points];
            direction_a[0] = 0.0; direction_a[1] = 0.0; direction_a[2] = 1.0;
            p0 = p0 - length_a*direction_a;
            for(unsigned int i=0; i<NEvaluations; i++)
            {
                point = p0 + i*(2.0*length_a/NEvaluations)*direction_a;
                x_coord.AddNextValue(point[0]);
                y_coord.AddNextValue(point[1]);
                z_coord.AddNextValue(point[2]);
                points[i] = point;
            }
        break;
        case 4:
            n_points = NEvaluations*NEvaluations;
            points = new KEMThreeVector[n_points];
            direction_a[0] = 0.0; direction_a[1] = 1.0; direction_a[2] = 0.0;
            direction_b[0] = 0.0; direction_b[1] = 0.0; direction_b[2] = 1.0;
            p0 = p0 - length_a*direction_a;
            p0 = p0 - length_b*direction_b;
            for(unsigned int i=0; i<NEvaluations; i++)
            {
                for(unsigned int j=0; j<NEvaluations; j++)
                {
                    point = p0 + i*(2.0*length_a/NEvaluations)*direction_a + j*(2.0*length_b/NEvaluations)*direction_b;
                    x_coord.AddNextValue(point[0]);
                    y_coord.AddNextValue(point[1]);
                    z_coord.AddNextValue(point[2]);
                    points[ j + NEvaluations*i ] = point;
                }
            }
        break;
        case 5:
            n_points = NEvaluations*NEvaluations;
            points = new KEMThreeVector[n_points];
            direction_a[0] = 1.0; direction_a[1] = 0.0; direction_a[2] = 0.0;
            direction_b[0] = 0.0; direction_b[1] = 0.0; direction_b[2] = 1.0;
            p0 = p0 - length_a*direction_a;
            p0 = p0 - length_b*direction_b;
            for(unsigned int i=0; i<NEvaluations; i++)
            {
                for(unsigned int j=0; j<NEvaluations; j++)
                {
                    point = p0 + i*(2.0*length_a/NEvaluations)*direction_a + j*(2.0*length_b/NEvaluations)*direction_b;
                    x_coord.AddNextValue(point[0]);
                    y_coord.AddNextValue(point[1]);
                    z_coord.AddNextValue(point[2]);
                    points[ j + NEvaluations*i ] = point;
                }
            }
        break;
        case 6:
            n_points = NEvaluations*NEvaluations;
            points = new KEMThreeVector[n_points];
            direction_a[0] = 1.0; direction_a[1] = 0.0; direction_a[2] = 0.0;
            direction_b[0] = 0.0; direction_b[1] = 1.0; direction_b[2] = 0.0;
            p0 = p0 - length_a*direction_a;
            p0 = p0 - length_b*direction_b;
            for(unsigned int i=0; i<NEvaluations; i++)
            {
                for(unsigned int j=0; j<NEvaluations; j++)
                {
                    point = p0 + i*(2.0*length_a/NEvaluations)*direction_a + j*(2.0*length_b/NEvaluations)*direction_b;
                    x_coord.AddNextValue(point[0]);
                    y_coord.AddNextValue(point[1]);
                    z_coord.AddNextValue(point[2]);
                    points[ j + NEvaluations*i ] = point;
                }
            }
        break;
        case 7:
            n_points = NEvaluations*NEvaluations*NEvaluations;
            points = new KEMThreeVector[n_points];
            direction_a[0] = 1.0; direction_a[1] = 0.0; direction_a[2] = 0.0;
            direction_b[0] = 0.0; direction_b[1] = 1.0; direction_b[2] = 0.0;
            direction_c[0] = 0.0; direction_c[1] = 0.0; direction_c[2] = 1.0;
            p0 = p0 - length_a*direction_a;
            p0 = p0 - length_b*direction_b;
            p0 = p0 - length_c*direction_c;
            for(unsigned int i=0; i<NEvaluations; i++)
            {
                for(unsigned int j=0; j<NEvaluations; j++)
                {
                    for(unsigned int k=0; k<NEvaluations; k++)
                    {
                        point = p0 + i*(2.0*length_a/NEvaluations)*direction_a + j*(2.0*length_b/NEvaluations)*direction_b + k*(2.0*length_c/NEvaluations)*direction_c;
                        x_coord.AddNextValue(point[0]);
                        y_coord.AddNextValue(point[1]);
                        z_coord.AddNextValue(point[2]);
                        points[ k + NEvaluations*(j + NEvaluations*i) ] = point;
                    }
                }
            }
        break;
    }


    std::cout<<std::setprecision(7);

    std::cout<<"starting potential/field evaluation"<<std::endl;

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
        KEMThreeVector field = fast_solver->ElectricField(points[i]);
        //std::cout<<"field = "<<field[0]<<", "<<field[1]<<", "<<field[2]<<std::endl;
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
        KEMThreeVector field = direct_solver->ElectricField(points[i]);
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

    std::cout<<"done computing errors"<<std::endl;


    //get tree/call data
    for(unsigned int i=0; i<n_points; i++)
    {
        tree_level.AddNextValue(fast_solver->GetTreeLevel(points[i]));
        n_direct_calls.AddNextValue(fast_solver->GetSubsetSize(points[i]));
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
    data_collection.AddData(tree_level);
    data_collection.AddData(n_direct_calls);

    //timing data
    data_collection.AddData(fmm_time_per_potential_call);
    data_collection.AddData(fmm_time_per_field_call);
    data_collection.AddData(direct_time_per_potential_call);
    data_collection.AddData(direct_time_per_field_call);



    KSAObjectOutputNode< KFMNamedScalarDataCollection >* data = new KSAObjectOutputNode< KFMNamedScalarDataCollection >("data_collection");
    data->AttachObjectToNode(&data_collection);

    bool result;
    KEMFileInterface::GetInstance()->SaveKSAFileToActiveDirectory(data, data_outfile, result, true);


    delete proxy_surfaceContainer;

    return 0;

  }
