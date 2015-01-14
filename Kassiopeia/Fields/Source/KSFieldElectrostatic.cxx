#include "KSFieldElectrostatic.h"
#include "KSFieldsMessage.h"

#ifdef KEMFIELD_USE_VTK
#include "KEMVTKViewer.hh"
#include "KEMVTKElectromagnetViewer.hh"
#include "KVTKIterationPlotter.hh"
using KEMField::KEMVTKViewer;
using KEMField::KEMVTKElectromagnetViewer;
using KEMField::KVTKIterationPlotter;
#endif

#include "KFile.h"

#include "KSADataStreamer.hh"

namespace Kassiopeia
{

    const unsigned int KSFieldElectrostatic::sNoSymmetry = 0;
    const unsigned int KSFieldElectrostatic::sAxialSymmetry = 1;
    const unsigned int KSFieldElectrostatic::sDiscreteAxialSymmetry = 2;

    KSFieldElectrostatic::KSFieldElectrostatic() :
            fDirectory( KEMFileInterface::GetInstance()->ActiveDirectory() ),
            fHashMaskedBits( 20 ),
            fHashThreshold( 1.e-14 ),
            fFile(),
            fSystem( NULL ),
            fSurfaces(),
            fSpaces(),
            fSymmetry( sNoSymmetry ),
            fConverter( NULL ),
            fContainer( NULL ),
            fBEMSolver( NULL ),
            fFieldSolver( NULL )
    {
        fFile = KEMFileInterface::GetInstance()->GetActiveFileName();
        fFile = fFile.substr( fFile.find_last_of( "/" ) + 1, std::string::npos );
    }
    KSFieldElectrostatic::KSFieldElectrostatic( const KSFieldElectrostatic& /*aCopy*/) :
            fDirectory( KEMFileInterface::GetInstance()->ActiveDirectory() ),
            fHashMaskedBits( 20 ),
            fHashThreshold( 1.e-14 ),
            fFile(),
            fSystem( NULL ),
            fSurfaces(),
            fSpaces(),
            fSymmetry( sNoSymmetry ),
            fConverter( NULL ),
            fContainer( NULL ),
            fBEMSolver( NULL ),
            fFieldSolver( NULL )
    {
        fFile = KEMFileInterface::GetInstance()->GetActiveFileName();
        fFile = fFile.substr( fFile.find_last_of( "/" ) + 1, std::string::npos );
    }
    KSFieldElectrostatic* KSFieldElectrostatic::Clone() const
    {
        return new KSFieldElectrostatic( *this );
    }
    KSFieldElectrostatic::~KSFieldElectrostatic()
    {
        for (vector<Visitor*>::iterator it = fVisitors.begin();it != fVisitors.end();++it)
          delete (*it);
        delete fBEMSolver;
        delete fFieldSolver;
        delete fContainer;
        delete fConverter;
    }

    void KSFieldElectrostatic::CalculatePotential( const KThreeVector& aSamplePoint, const double& /*aSampleTime*/, double& aPotential )
    {
        aPotential = fFieldSolver->Potential( fConverter->GlobalToInternalPosition( aSamplePoint ) );
        fieldmsg_debug( "potential at " <<aSamplePoint<< " is " << aPotential <<eom);
        return;
    }

    void KSFieldElectrostatic::CalculateField( const KThreeVector& aSamplePoint, const double& /*aSampleTime*/, KThreeVector& aField )
    {
        aField = fConverter->InternalToGlobalVector( fFieldSolver->ElectricField( fConverter->GlobalToInternalPosition( aSamplePoint ) ) );
        fieldmsg_debug( "electric field at " <<aSamplePoint<< " is " << aField <<eom);
        return;
    }
    void KSFieldElectrostatic::SetDirectory( const string& aDirectory )
    {
        fDirectory = aDirectory;
        return;
    }
    void KSFieldElectrostatic::SetFile( const string& aFile )
    {
        fFile = aFile;
        return;
    }
    void KSFieldElectrostatic::SetSystem( KGSpace* aSpace )
    {
        fSystem = aSpace;
        return;
    }
    void KSFieldElectrostatic::AddSurface( KGSurface* aSurface )
    {
        fSurfaces.push_back( aSurface );
        return;
    }
    void KSFieldElectrostatic::AddSpace( KGSpace* aSpace )
    {
        fSpaces.push_back( aSpace );
        return;
    }
    void KSFieldElectrostatic::SetSymmetry( const unsigned int& aSymmetry )
    {
        fSymmetry = aSymmetry;
        return;
    }
    void KSFieldElectrostatic::SetHashMaskedBits( const unsigned int& aMaskedBits )
    {
        fHashMaskedBits = aMaskedBits;
        return;
    }
    void KSFieldElectrostatic::SetHashThreshold( const double& aThreshold )
    {
        fHashThreshold = aThreshold;
        return;
    }

    //**********
    //visitor
    //**********

    KSFieldElectrostatic::Visitor::Visitor() :
      fPreprocessing( false ),
      fPostprocessing( false )
    {
    }

    //**********
    //vtk viewer
    //**********

    KSFieldElectrostatic::VTKViewer::VTKViewer() :
      fViewGeometry( false ),
      fSaveGeometry( false ),
      fFile( "ElectrostaticGeometry.vtp" )
    {
    }
    KSFieldElectrostatic::VTKViewer::~VTKViewer()
    {
    }
    void KSFieldElectrostatic::VTKViewer::Visit(KSFieldElectrostatic& electrostaticFieldSolver)
    {
#ifdef KEMFIELD_USE_VTK
      if (fViewGeometry || fSaveGeometry)
      {
        KEMVTKViewer viewer(*(electrostaticFieldSolver.GetContainer()));
        if (fViewGeometry)
          viewer.ViewGeometry();
        if (fSaveGeometry)
        {
          KEMField::cout<<"Saving electrode geometry to "<<fFile<<"."<<KEMField::endl;
          viewer.GenerateGeometryFile(fFile);
        }
      }
#else
        (void)electrostaticFieldSolver;
#endif
    }

    //**********
    //bem solver
    //**********

    KSFieldElectrostatic::BEMSolver::BEMSolver() :
            fHashMaskedBits( 20 ),
            fHashThreshold( 1.e-14 )
    {
    }
    KSFieldElectrostatic::BEMSolver::~BEMSolver()
    {
    }
    void KSFieldElectrostatic::BEMSolver::SetHashProperties( unsigned int maskedBits, double hashThreshold )
    {
        fHashMaskedBits = maskedBits;
        fHashThreshold = hashThreshold;
    }
    bool KSFieldElectrostatic::BEMSolver::FindSolution( double aThreshold, KSurfaceContainer& aContainer )
    {
        // compute shape hash
        KMD5HashGenerator tShapeHashGenerator;
        tShapeHashGenerator.MaskedBits( fHashMaskedBits );
        tShapeHashGenerator.Threshold( fHashThreshold );
        tShapeHashGenerator.Omit( KEMField::Type2Type< KElectrostaticBasis >() );
        tShapeHashGenerator.Omit( Type2Type< KBoundaryType< KElectrostaticBasis, KDirichletBoundary > >() );
        string tShapeHash = tShapeHashGenerator.GenerateHash( aContainer );

        fieldmsg_debug( "<shape> hash is <" << tShapeHash << ">" << eom )

        // compute shape+boundary hash
        KMD5HashGenerator tShapeBoundaryHashGenerator;
        tShapeBoundaryHashGenerator.MaskedBits( fHashMaskedBits );
        tShapeBoundaryHashGenerator.Threshold( fHashThreshold );
        tShapeBoundaryHashGenerator.Omit( Type2Type< KElectrostaticBasis >() );
        string tShapeBoundaryHash = tShapeBoundaryHashGenerator.GenerateHash( aContainer );

        fieldmsg_debug( "<shape+boundary> hash is <" << tShapeBoundaryHash << ">" << eom )

        vector< string > tLabels;
        unsigned int tCount;
        bool tSolution;

        // compose residual threshold labels for shape and shape+boundary
        tLabels.clear();
        tLabels.push_back( KResidualThreshold::Name() );
        tLabels.push_back( tShapeHash );
        tLabels.push_back( tShapeBoundaryHash );

        // find matching residual thresholds
        tCount = KEMFileInterface::GetInstance()->NumberWithLabels( tLabels );

        fieldmsg_debug( "found <" << tCount << "> that match <shape> and <shape+boundary> hashes" << eom )

        if( tCount > 0 )
        {
            KResidualThreshold tResidualThreshold;
            KResidualThreshold tMinResidualThreshold;

            for( unsigned int i = 0; i < tCount; i++ )
            {
                KEMFileInterface::GetInstance()->FindByLabels( tResidualThreshold, tLabels, i );

                fieldmsg_debug( "found threshold <" << tResidualThreshold.fResidualThreshold << ">" << eom )

                if( tResidualThreshold < tMinResidualThreshold )
                {
                    fieldmsg_debug( "found minimum solution <" << tResidualThreshold.fGeometryHash << "> with threshold <" << tResidualThreshold.fResidualThreshold << ">" << eom )

                    tMinResidualThreshold = tResidualThreshold;
                }

            }

            fieldmsg_debug( "global minimum solution <" << tMinResidualThreshold.fGeometryHash << "> with threshold <" << tResidualThreshold.fResidualThreshold << ">" << eom )

            KEMFileInterface::GetInstance()->FindByHash( aContainer, tMinResidualThreshold.fGeometryHash );

            tSolution = false;
            if( tMinResidualThreshold.fResidualThreshold <= aThreshold )
            {
                MPI_SINGLE_PROCESS
                    fieldmsg( eNormal ) << "previously computed solution found" << eom;
                tSolution = true;
            }

            if( tSolution == true )
            {
                return true;
            }
        }

        // compose residual threshold labels for shape
        tLabels.clear();
        tLabels.push_back( KResidualThreshold::Name() );
        tLabels.push_back( tShapeHash );

        // find residual thresholds for geometry
        tCount = KEMFileInterface::GetInstance()->NumberWithLabels( tLabels );

        fieldmsg_debug( "found <" << tCount << "> that match <shape> hash" << eom )

        if( tCount > 0 )
        {
#ifdef KEMFIELD_USE_ROOT
            KSuperpositionSolver< double, KEMRootSVDSolver > tSuperpositionSolver;
#else
            KSuperpositionSolver< double, KSVDSolver > tSuperpositionSolver; // this doesn't seem to work even if gsl is enabled
#endif

            KElectrostaticBoundaryIntegrator tIntegrator;
            KBoundaryIntegralVector< KElectrostaticBoundaryIntegrator > tVector( aContainer, tIntegrator );
            KBoundaryIntegralSolutionVector< KElectrostaticBoundaryIntegrator > tSolutionVector( aContainer, tIntegrator );

            KResidualThreshold tResidualThreshold;
            vector< KSurfaceContainer* > tContainers;
            vector< KBoundaryIntegralSolutionVector< KElectrostaticBoundaryIntegrator >* > tSolutionVectors;
            vector< KBoundaryIntegralVector< KElectrostaticBoundaryIntegrator >* > tVectors;

            for( unsigned int tIndex = 0; tIndex < tCount; tIndex++ )
            {
                KEMFileInterface::GetInstance()->FindByLabels( tResidualThreshold, tLabels, tIndex );

                fieldmsg_debug( "found threshold <" << tResidualThreshold.fResidualThreshold << ">" << eom )

                if( tResidualThreshold.fResidualThreshold <= aThreshold )
                {
                    fieldmsg_debug( "adding solution <" << tResidualThreshold.fGeometryHash << "> with threshold <" << tResidualThreshold.fResidualThreshold << ">" << eom )

                    KSurfaceContainer* tNewContainer = new KSurfaceContainer();
                    KEMFileInterface::GetInstance()->FindByHash( *tNewContainer, tResidualThreshold.fGeometryHash );

                    KBoundaryIntegralVector< KElectrostaticBoundaryIntegrator >* tNewVector = new KBoundaryIntegralVector< KElectrostaticBoundaryIntegrator >( *tNewContainer, tIntegrator );
                    KBoundaryIntegralSolutionVector< KElectrostaticBoundaryIntegrator >* tNewSolutionVector = new KBoundaryIntegralSolutionVector< KElectrostaticBoundaryIntegrator >( *tNewContainer, tIntegrator );

                    tContainers.push_back( tNewContainer );
                    tVectors.push_back( tNewVector );
                    tSolutionVectors.push_back( tNewSolutionVector );
                    tSuperpositionSolver.AddSolvedSystem( *tNewSolutionVector, *tNewVector );
                }
            }

            tSolution = false;
            if( tSuperpositionSolver.SolutionSpaceIsSpanned( tVector ) )
            {
                tSuperpositionSolver.ComposeSolution( tSolutionVector );
                fieldmsg( eNormal ) << "superposition of previously computed solutions found" << eom;
                tSolution = true;
            }

            for( unsigned int i = 0; i < tContainers.size(); i++ )
            {
                delete tContainers.at( i );
            }
            for( unsigned int i = 0; i < tSolutionVectors.size(); i++ )
            {
                delete tSolutionVectors.at( i );
            }
            for( unsigned int i = 0; i < tVectors.size(); i++ )
            {
                delete tVectors.at( i );
            }

            if( tSolution == true )
            {
                return true;
            }
        }
        return false;
    }
    void KSFieldElectrostatic::BEMSolver::SaveSolution( double aThreshold, KSurfaceContainer& aContainer )
    {
        // compute hash of the bare geometry
        KMD5HashGenerator tShapeHashGenerator;
        tShapeHashGenerator.MaskedBits( fHashMaskedBits );
        tShapeHashGenerator.Threshold( fHashThreshold );
        tShapeHashGenerator.Omit( Type2Type< KElectrostaticBasis >() );
        tShapeHashGenerator.Omit( Type2Type< KBoundaryType< KElectrostaticBasis, KDirichletBoundary > >() );
        string tShapeHash = tShapeHashGenerator.GenerateHash( aContainer );

        fieldmsg_debug( "<shape> hash is <" << tShapeHash << ">" << eom )

        // compute hash of the boundary values on the bare geometry
        KMD5HashGenerator tShapeBoundaryHashGenerator;
        tShapeBoundaryHashGenerator.MaskedBits( fHashMaskedBits );
        tShapeBoundaryHashGenerator.Threshold( fHashThreshold );
        tShapeBoundaryHashGenerator.Omit( Type2Type< KElectrostaticBasis >() );
        string tShapeBoundaryHash = tShapeBoundaryHashGenerator.GenerateHash( aContainer );

        fieldmsg_debug( "<shape+boundary> hash is <" << tShapeBoundaryHash << ">" << eom )

        // compute hash of solution with boundary values on the bare geometry
        KMD5HashGenerator tShapeBoundarySolutionHashGenerator;
        string tShapeBoundarySolutionHash = tShapeBoundarySolutionHashGenerator.GenerateHash( aContainer );

        fieldmsg_debug( "<shape+boundary+solution> hash is <" << tShapeBoundarySolutionHash << ">" << eom )

        // create label set for summary object
        string tThresholdBase( KResidualThreshold::Name() );
        string tThresholdName = tThresholdBase + string( "_" ) + tShapeBoundarySolutionHash;
        vector< string > tThresholdLabels;
        tThresholdLabels.push_back( tThresholdBase );
        tThresholdLabels.push_back( tShapeHash );
        tThresholdLabels.push_back( tShapeBoundaryHash );
        tThresholdLabels.push_back( tShapeBoundarySolutionHash );

        // write summary object;
        KResidualThreshold tResidualThreshold;
        tResidualThreshold.fResidualThreshold = aThreshold;
        tResidualThreshold.fGeometryHash = tShapeBoundarySolutionHash;
        KEMFileInterface::GetInstance()->Write( tResidualThreshold, tThresholdName, tThresholdLabels );

        // create label set for container object
        string tContainerBase( KSurfaceContainer::Name() );
        string tContainerName = tContainerBase + string( "_" ) + tShapeBoundarySolutionHash;
        vector< string > tContainerLabels;
        tContainerLabels.push_back( tContainerBase );
        tContainerLabels.push_back( tShapeBoundarySolutionHash );

        // write container object
        KEMFileInterface::GetInstance()->Write( aContainer, tContainerName, tContainerLabels );

        return;
    }

    //*****************
    //cached bem solver
    //*****************

    KSFieldElectrostatic::CachedBEMSolver::CachedBEMSolver() :
            fName(),
            fHash()
    {
    }
    KSFieldElectrostatic::CachedBEMSolver::~CachedBEMSolver()
    {
    }
    void KSFieldElectrostatic::CachedBEMSolver::Initialize( KSurfaceContainer& container )
    {
        bool tSolution = false;

        if( (fName.size() == 0) && (fHash.size() == 0) )
        {
            fieldmsg( eError ) << "must provide a name or a hash for cached bem solution" << eom;
        }

        if( fName.size() != 0 )
        {
            KEMFileInterface::GetInstance()->FindByName( container, fName, tSolution );
        }
        else if( fHash.size() != 0 )
        {
            KEMFileInterface::GetInstance()->FindByHash( container, fHash, tSolution );
        }

        if( tSolution == false )
        {
            fieldmsg << "could not find cached bem solution in directory <" << KEMFileInterface::GetInstance()->ActiveDirectory() << ">" << ret;
            fieldmsg << "with name <" << fName << "> and hash <" << fHash << ">";
            fieldmsg( eError ) << eom;
        }
    }

    //*******************************
    //gaussian elimination bem solver
    //*******************************

    KSFieldElectrostatic::GaussianEliminationBEMSolver::GaussianEliminationBEMSolver()
    {
    }
    KSFieldElectrostatic::GaussianEliminationBEMSolver::~GaussianEliminationBEMSolver()
    {
    }
    void KSFieldElectrostatic::GaussianEliminationBEMSolver::Initialize( KSurfaceContainer& container )
    {
        if( FindSolution( 0., container ) == false )
        {
            KElectrostaticBoundaryIntegrator integrator;
            KBoundaryIntegralMatrix< KElectrostaticBoundaryIntegrator > A( container, integrator );
            KBoundaryIntegralSolutionVector< KElectrostaticBoundaryIntegrator > x( container, integrator );
            KBoundaryIntegralVector< KElectrostaticBoundaryIntegrator > b( container, integrator );

            KGaussianElimination< KElectrostaticBoundaryIntegrator::ValueType > gaussianElimination;
            gaussianElimination.Solve( A, x, b );

            SaveSolution( 0., container );
        }
        return;
    }

    //*********************
    //robin hood bem solver
    //*********************

    KSFieldElectrostatic::RobinHoodBEMSolver::RobinHoodBEMSolver() :
            fTolerance( 1.e-8 ),
            fCheckSubInterval( 100 ),
            fDisplayInterval( 0 ),
            fWriteInterval( 0 ),
            fPlotInterval( 0 ),
            fCacheMatrixElements( false ),
            fUseOpenCL( false ),
            fUseVTK( false )
    {
    }
    KSFieldElectrostatic::RobinHoodBEMSolver::~RobinHoodBEMSolver()
    {
#ifdef KEMFIELD_USE_OPENCL
        if( fUseOpenCL )
        {
            KOpenCLSurfaceContainer* oclContainer = dynamic_cast< KOpenCLSurfaceContainer* >( KOpenCLInterface::GetInstance()->GetActiveData() );
            if( oclContainer )
                delete oclContainer;
            oclContainer = NULL;
            KOpenCLInterface::GetInstance()->SetActiveData( oclContainer );
        }
#endif
    }
    void KSFieldElectrostatic::RobinHoodBEMSolver::Initialize( KSurfaceContainer& container )
    {
        if( FindSolution( fTolerance, container ) == false )
        {
            if( fUseOpenCL )
            {
#if defined(KEMFIELD_USE_MPI) && defined(KEMFIELD_USE_OPENCL)
                KOpenCLInterface::GetInstance()->SetGPU(KMPIInterface::GetInstance()->GetProcess());
#endif

#ifdef KEMFIELD_USE_OPENCL
                KOpenCLSurfaceContainer* oclContainer = new KOpenCLSurfaceContainer( container );
                KOpenCLInterface::GetInstance()->SetActiveData( oclContainer );
                KOpenCLElectrostaticBoundaryIntegrator integrator( *oclContainer );
                KBoundaryIntegralMatrix< KOpenCLBoundaryIntegrator< KElectrostaticBasis > > A( *oclContainer, integrator );
                KBoundaryIntegralVector< KOpenCLBoundaryIntegrator< KElectrostaticBasis > > b( *oclContainer, integrator );
                KBoundaryIntegralSolutionVector< KOpenCLBoundaryIntegrator< KElectrostaticBasis > > x( *oclContainer, integrator );

#ifdef KEMFIELD_USE_MPI
                KRobinHood< KElectrostaticBoundaryIntegrator::ValueType, KRobinHood_MPI_OpenCL > robinHood;
#else
                KRobinHood< KElectrostaticBoundaryIntegrator::ValueType, KRobinHood_OpenCL > robinHood;
#endif
                robinHood.SetTolerance( fTolerance );
                robinHood.SetResidualCheckInterval( fCheckSubInterval );

                if( fDisplayInterval != 0 )
                {
                    MPI_SINGLE_PROCESS
                    {
                        KIterationDisplay< KElectrostaticBoundaryIntegrator::ValueType >* display = new KIterationDisplay< KElectrostaticBoundaryIntegrator::ValueType >();
                        display->Interval( fDisplayInterval );
                        robinHood.AddVisitor( display );
                    }
                }
                if( fWriteInterval != 0 )
                {
                    KIterativeStateWriter< KElectrostaticBoundaryIntegrator::ValueType >* stateWriter = new KIterativeStateWriter< KElectrostaticBoundaryIntegrator::ValueType >( container );
                    stateWriter->Interval( fWriteInterval );
                    robinHood.AddVisitor( stateWriter );
                }
                if( fPlotInterval != 0 )
                {
                    if( fUseVTK == true )
                    {
#ifdef KEMFIELD_USE_VTK
                        MPI_SINGLE_PROCESS
                        {
                            KVTKIterationPlotter< KElectrostaticBoundaryIntegrator::ValueType >* plotter = new KVTKIterationPlotter< KElectrostaticBoundaryIntegrator::ValueType >();
                            plotter->Interval( fPlotInterval );
                            robinHood.AddVisitor( plotter );
                        }
#endif
                    }
                }

                robinHood.Solve( A, x, b );

                MPI_SINGLE_PROCESS
                {
                    SaveSolution( fTolerance, container );
                }
                return;
#endif
            }
            KElectrostaticBoundaryIntegrator integrator;
            KSquareMatrix< double > *A;
            if( fCacheMatrixElements )
                A = new KBoundaryIntegralMatrix< KElectrostaticBoundaryIntegrator, true >( container, integrator );
            else
                A = new KBoundaryIntegralMatrix< KElectrostaticBoundaryIntegrator >( container, integrator );
            KBoundaryIntegralSolutionVector< KElectrostaticBoundaryIntegrator > x( container, integrator );
            KBoundaryIntegralVector< KElectrostaticBoundaryIntegrator > b( container, integrator );

#ifdef KEMFIELD_USE_MPI
            KRobinHood< KElectrostaticBoundaryIntegrator::ValueType, KRobinHood_MPI > robinHood;
#else
            KRobinHood< KElectrostaticBoundaryIntegrator::ValueType > robinHood;
#endif
            robinHood.SetTolerance( fTolerance );
            robinHood.SetResidualCheckInterval( fCheckSubInterval );

            if( fDisplayInterval != 0 )
            {
                MPI_SINGLE_PROCESS
                {
                    KIterationDisplay< KElectrostaticBoundaryIntegrator::ValueType >* display = new KIterationDisplay< KElectrostaticBoundaryIntegrator::ValueType >();
                    display->Interval( fDisplayInterval );
                    robinHood.AddVisitor( display );
                }
            }
            if( fWriteInterval != 0 )
            {
                MPI_SINGLE_PROCESS
                {
                    KIterativeStateWriter< KElectrostaticBoundaryIntegrator::ValueType >* stateWriter = new KIterativeStateWriter< KElectrostaticBoundaryIntegrator::ValueType >( container );
                    stateWriter->Interval( fWriteInterval );
                    robinHood.AddVisitor( stateWriter );
                }
            }
            if( fPlotInterval != 0 )
            {
                if( fUseVTK == true )
                {
#ifdef KEMFIELD_USE_VTK
                    MPI_SINGLE_PROCESS
                    {
                        KVTKIterationPlotter< KElectrostaticBoundaryIntegrator::ValueType >* plotter = new KVTKIterationPlotter< KElectrostaticBoundaryIntegrator::ValueType >();
                        plotter->Interval( fPlotInterval );
                        robinHood.AddVisitor( plotter );
                    }
#endif
                }
            }

            robinHood.Solve( *A, x, b );

            delete A;

            MPI_SINGLE_PROCESS
            {
                SaveSolution( fTolerance, container );
            }
            return;
        }
    }

    //*************************
    //fast multipole bem solver
    //*************************

KSFieldElectrostatic::FastMultipoleBEMSolver::FastMultipoleBEMSolver() :
            fTolerance( 1.e-8 ),
            fParameters(),
            fKrylov("gmres"),
            fRestartCycle(30),
            fUseOpenCL(false),
            fUseVTK(false){};

    KSFieldElectrostatic::FastMultipoleBEMSolver::~FastMultipoleBEMSolver(){};

    void KSFieldElectrostatic::FastMultipoleBEMSolver::Initialize( KSurfaceContainer& surfaceContainer )
    {
        if( FindSolution( fTolerance, surfaceContainer ) == false )
        {

            //now we want to construct the tree
            KFMElectrostaticParameters params = fParameters;
            params.degree = 0; //mask the expansion degree, as this does not affect disk-cached matrix elements

            // compute hash of the bare geometry
            KMD5HashGenerator tShapeHashGenerator;
            tShapeHashGenerator.MaskedBits( fHashMaskedBits );
            tShapeHashGenerator.Threshold( fHashThreshold );
            tShapeHashGenerator.Omit( Type2Type< KElectrostaticBasis >() );
            tShapeHashGenerator.Omit( Type2Type< KBoundaryType< KElectrostaticBasis, KDirichletBoundary > >() );
            string tShapeHash = tShapeHashGenerator.GenerateHash( surfaceContainer );

            // compute hash of the parameter values
            KMD5HashGenerator parameterHashGenerator;
            parameterHashGenerator.MaskedBits( fHashMaskedBits );
            parameterHashGenerator.Threshold( fHashThreshold );
            string parameterHash = parameterHashGenerator.GenerateHash( params );

            //construct a unique id by stripping the first 6 characters from the shape and parameter hashes
            //this id is only used to append to the names of cached sparse matrix element files
            std::string unique_id = tShapeHash.substr(0,6) + parameterHash.substr(0,6);

            if( fUseOpenCL )
            {
                #ifndef KEMFIELD_USE_OPENCL
                    typedef KFMElectrostaticBoundaryIntegrator<KFMElectrostaticBoundaryIntegratorEngine_SingleThread> FastMultipoleEBI;
                #else
                    typedef KFMElectrostaticBoundaryIntegrator<KFMElectrostaticBoundaryIntegratorEngine_OpenCL> FastMultipoleEBI;
                #endif

                FastMultipoleEBI integrator(surfaceContainer);
                integrator.SetUniqueIDString(unique_id);
                integrator.Initialize(fParameters);
                KFMBoundaryIntegralMatrix< FastMultipoleEBI > A(surfaceContainer,integrator);
                KBoundaryIntegralSolutionVector< FastMultipoleEBI > x(surfaceContainer,integrator);
                KBoundaryIntegralVector< FastMultipoleEBI > b(surfaceContainer,integrator);

                if( fKrylov == std::string("gmres") )
                {
                    KIterativeKrylovSolver< FastMultipoleEBI::ValueType, KGeneralizedMinimalResidual> gmres;
                    gmres.SetTolerance(fTolerance);
                    KIterativeKrylovRestartCondition* restart_condition = new KIterativeKrylovRestartCondition();
                    restart_condition->SetNumberOfIterationsBetweenRestart(fRestartCycle);
                    gmres.SetRestartCondition(restart_condition);


                    gmres.AddVisitor(new KIterationDisplay<double>());

                    #ifdef KEMFIELD_USE_VTK
                    if( fUseVTK )
                    {
                        gmres.AddVisitor(new KVTKIterationPlotter<double>());
                    }
                    #endif

                    gmres.Solve(A,x,b);
                    delete restart_condition;
                }
                else if( fKrylov == std::string("bicgstab") )
                {
                    KIterativeKrylovSolver< FastMultipoleEBI::ValueType, KBiconjugateGradientStabilized> bicgstab;
                    bicgstab.SetTolerance(fTolerance);

                    bicgstab.AddVisitor(new KIterationDisplay<double>());

                    #ifdef KEMFIELD_USE_VTK
                    if( fUseVTK )
                    {
                        bicgstab.AddVisitor(new KVTKIterationPlotter<double>());
                    }
                    #endif

                    bicgstab.Solve(A,x,b);
                }
            }
            else
            {
                typedef KFMElectrostaticBoundaryIntegrator<KFMElectrostaticBoundaryIntegratorEngine_SingleThread> SingleThread_FastMultipoleEBI;

                SingleThread_FastMultipoleEBI integrator(surfaceContainer);
                integrator.SetUniqueIDString(unique_id);
                integrator.Initialize(fParameters);
                KFMBoundaryIntegralMatrix< SingleThread_FastMultipoleEBI > A(surfaceContainer,integrator);
                KBoundaryIntegralSolutionVector< SingleThread_FastMultipoleEBI > x(surfaceContainer,integrator);
                KBoundaryIntegralVector< SingleThread_FastMultipoleEBI > b(surfaceContainer,integrator);

                if( fKrylov == std::string("gmres") )
                {
                    KIterativeKrylovSolver< SingleThread_FastMultipoleEBI::ValueType, KGeneralizedMinimalResidual> gmres;
                    gmres.SetTolerance(fTolerance);

                    KIterativeKrylovRestartCondition* restart_condition = new KIterativeKrylovRestartCondition();
                    restart_condition->SetNumberOfIterationsBetweenRestart(fRestartCycle);
                    gmres.SetRestartCondition(restart_condition);

                    gmres.AddVisitor(new KIterationDisplay<double>());

                    #ifdef KEMFIELD_USE_VTK
                    if( fUseVTK )
                    {
                        gmres.AddVisitor(new KVTKIterationPlotter<double>());
                    }
                    #endif

                    gmres.Solve(A,x,b);
                    delete restart_condition;
                }
                else if( fKrylov == std::string("bicgstab") )
                {
                    KIterativeKrylovSolver< SingleThread_FastMultipoleEBI::ValueType, KBiconjugateGradientStabilized> bicgstab;
                    bicgstab.SetTolerance(fTolerance);

                    bicgstab.AddVisitor(new KIterationDisplay<double>());

                    #ifdef KEMFIELD_USE_VTK
                    if( fUseVTK )
                    {
                        bicgstab.AddVisitor(new KVTKIterationPlotter<double>());
                    }
                    #endif

                    bicgstab.Solve(A,x,b);
                }
            }

            SaveSolution( fTolerance, surfaceContainer );
        }
    }

    //************
    //field solver
    //************

    KSFieldElectrostatic::FieldSolver::FieldSolver()
    {
    }
    KSFieldElectrostatic::FieldSolver::~FieldSolver()
    {
    }
    KSFieldElectrostatic::IntegratingFieldSolver::IntegratingFieldSolver() :
            fIntegrator( NULL ),
            fIntegratingFieldSolver( NULL ),
#ifdef KEMFIELD_USE_OPENCL
            fOCLIntegrator( NULL ),
            fOCLIntegratingFieldSolver( NULL ),
#endif
            fUseOpenCL( false )
    {

    }

    KSFieldElectrostatic::IntegratingFieldSolver::~IntegratingFieldSolver()
    {
        delete fIntegrator;
        delete fIntegratingFieldSolver;
#ifdef KEMFIELD_USE_OPENCL
        delete fOCLIntegrator;
        delete fOCLIntegratingFieldSolver;

        if( fUseOpenCL )
        {
            KOpenCLSurfaceContainer* oclContainer = dynamic_cast< KOpenCLSurfaceContainer* >( KOpenCLInterface::GetInstance()->GetActiveData() );
            if( oclContainer )
                delete oclContainer;
            oclContainer = NULL;
            KOpenCLInterface::GetInstance()->SetActiveData( oclContainer );
        }
#endif
    }

    void KSFieldElectrostatic::IntegratingFieldSolver::Initialize( KSurfaceContainer& container )
    {
        if( fUseOpenCL )
        {
#ifdef KEMFIELD_USE_OPENCL
            KOpenCLData* data = KOpenCLInterface::GetInstance()->GetActiveData();
            KOpenCLSurfaceContainer* oclContainer;
            if( data )
                oclContainer = dynamic_cast< KOpenCLSurfaceContainer* >( data );
            else
            {
                oclContainer = new KOpenCLSurfaceContainer( container );
                KOpenCLInterface::GetInstance()->SetActiveData( oclContainer );
            }
            fOCLIntegrator = new KOpenCLElectrostaticBoundaryIntegrator( *oclContainer );
            fOCLIntegratingFieldSolver = new KIntegratingFieldSolver< KOpenCLElectrostaticBoundaryIntegrator >( *oclContainer, *fOCLIntegrator );
            fOCLIntegratingFieldSolver->Initialize();
            return;
#endif
        }
        fIntegrator = new KElectrostaticBoundaryIntegrator();
        fIntegratingFieldSolver = new KIntegratingFieldSolver< KElectrostaticBoundaryIntegrator >( container, *fIntegrator );
    }

    double KSFieldElectrostatic::IntegratingFieldSolver::Potential( const KPosition& P ) const
    {
        if( fUseOpenCL )
        {
#ifdef KEMFIELD_USE_OPENCL
            return fOCLIntegratingFieldSolver->Potential( P );
#endif
        }
        return fIntegratingFieldSolver->Potential( P );
    }

    KEMThreeVector KSFieldElectrostatic::IntegratingFieldSolver::ElectricField( const KPosition& P ) const
    {
        if( fUseOpenCL )
        {
#ifdef KEMFIELD_USE_OPENCL
            return fOCLIntegratingFieldSolver->ElectricField( P );
#endif
        }
        return fIntegratingFieldSolver->ElectricField( P );
    }

    KSFieldElectrostatic::ZonalHarmonicFieldSolver::ZonalHarmonicFieldSolver() :
            fZHContainer( NULL ),
            fZonalHarmonicFieldSolver( NULL )
    {
        fParameters = new KZonalHarmonicParameters();
    }

    KSFieldElectrostatic::ZonalHarmonicFieldSolver::~ZonalHarmonicFieldSolver()
    {
        delete fZHContainer;
        delete fZonalHarmonicFieldSolver;
    }

    void KSFieldElectrostatic::ZonalHarmonicFieldSolver::Initialize( KSurfaceContainer& container )
    {
        // compute hash of the solved geometry
        KMD5HashGenerator solutionHashGenerator;
        string solutionHash = solutionHashGenerator.GenerateHash( container );

        fieldmsg_debug( "<shape+boundary+solution> hash is <" << solutionHash << ">" << eom )

        // compute hash of the parameter values on the bare geometry
        KMD5HashGenerator parameterHashGenerator;
        string parameterHash = parameterHashGenerator.GenerateHash( *fParameters );

        fieldmsg_debug( "<parameter> hash is <" << parameterHash << ">" << eom )

        // create label set for zh container object
        string zhContainerBase( KZonalHarmonicContainer< KElectrostaticBasis >::Name() );
        string zhContainerName = zhContainerBase + string( "_" ) + solutionHash + string( "_" ) + parameterHash;
        vector< string > zhContainerLabels;
        zhContainerLabels.push_back( zhContainerBase );
        zhContainerLabels.push_back( solutionHash );
        zhContainerLabels.push_back( parameterHash );

        fZHContainer = new KZonalHarmonicContainer< KElectrostaticBasis >( container, fParameters );

        bool containerFound = false;

        KEMFileInterface::GetInstance()->FindByLabels( *fZHContainer, zhContainerLabels, 0, containerFound );

        if( containerFound == true )
        {
            fieldmsg( eNormal ) << "zonal harmonic container found." << eom;
        }
        else
        {
            fieldmsg_debug( "no zonal harmonic container found." << eom )

            fZHContainer->ComputeCoefficients();

            KEMFileInterface::GetInstance()->Write( *fZHContainer, zhContainerName, zhContainerLabels );
        }

        fZonalHarmonicFieldSolver = new KZonalHarmonicFieldSolver< KElectrostaticBasis >( *fZHContainer, fIntegrator );
        fZonalHarmonicFieldSolver->Initialize();

        return;
    }

    double KSFieldElectrostatic::ZonalHarmonicFieldSolver::Potential( const KPosition& P ) const
    {
        return fZonalHarmonicFieldSolver->Potential( P );
    }

    KEMThreeVector KSFieldElectrostatic::ZonalHarmonicFieldSolver::ElectricField( const KPosition& P ) const
    {
        return fZonalHarmonicFieldSolver->ElectricField( P );
    }

    bool KSFieldElectrostatic::ZonalHarmonicFieldSolver::UseCentralExpansion(const KPosition &P)
    {
        return fZonalHarmonicFieldSolver->UseCentralExpansion( P );
    }

    bool KSFieldElectrostatic::ZonalHarmonicFieldSolver::UseRemoteExpansion(const KPosition &P)
    {
        return fZonalHarmonicFieldSolver->UseRemoteExpansion( P );
    }

    //***************************
    //fast multipole field solver
    //***************************

    KSFieldElectrostatic::FastMultipoleFieldSolver::FastMultipoleFieldSolver():
            fFastMultipoleFieldSolver( NULL ),
            #ifdef KEMFIELD_USE_OPENCL
            fFastMultipoleFieldSolverOpenCL(NULL),
            #endif
            fUseOpenCL(false)
    {
        fTree = new KFMElectrostaticTree();
    }

    KSFieldElectrostatic::FastMultipoleFieldSolver::~FastMultipoleFieldSolver()
    {
        delete fTree;
        delete fFastMultipoleFieldSolver;
        #ifdef KEMFIELD_USE_OPENCL
        delete fFastMultipoleFieldSolverOpenCL;

        if( fUseOpenCL )
        {
            KOpenCLSurfaceContainer* oclContainer = dynamic_cast< KOpenCLSurfaceContainer* >( KOpenCLInterface::GetInstance()->GetActiveData() );
            if( oclContainer )
                delete oclContainer;
            oclContainer = NULL;
            KOpenCLInterface::GetInstance()->SetActiveData( oclContainer );
        }

        #endif
    }

    void KSFieldElectrostatic::FastMultipoleFieldSolver::Initialize( KSurfaceContainer& container )
    {
        //the tree constuctor definitions
        typedef KFMElectrostaticTreeConstructor<KFMElectrostaticFieldMapper_SingleThread> TreeConstructor_SingleThread;
        #ifdef KEMFIELD_USE_OPENCL
        typedef KFMElectrostaticTreeConstructor< KFMElectrostaticFieldMapper_OpenCL > TreeConstructor_OpenCL;
        KOpenCLData* data = KOpenCLInterface::GetInstance()->GetActiveData();
        KOpenCLSurfaceContainer* oclContainer;
        if( data )
        {
            fieldmsg_debug( "using a prexisting OpenCL surface container." << eom );
            oclContainer = dynamic_cast< KOpenCLSurfaceContainer* >( data );
        }
        else
        {
            fieldmsg_debug( "creating a new OpenCL surface container." <<  eom );
            oclContainer = new KOpenCLSurfaceContainer( container );
            KOpenCLInterface::GetInstance()->SetActiveData( oclContainer );
        }
        #else
        typedef KFMElectrostaticTreeConstructor<KFMElectrostaticFieldMapper_SingleThread> TreeConstructor_OpenCL;
        #endif

        // compute hash of the solved geometry
        KMD5HashGenerator solutionHashGenerator;
        string solutionHash = solutionHashGenerator.GenerateHash( container );

        fieldmsg_debug( "<shape+boundary+solution> hash is <" << solutionHash << ">" << eom )

        // compute hash of the parameter values on the bare geometry
        // compute hash of the parameter values
        KMD5HashGenerator parameterHashGenerator;
        string parameterHash = parameterHashGenerator.GenerateHash( fParameters );

        fieldmsg_debug( "<parameter> hash is <" << parameterHash << ">" << eom );

        // create label set for multipole tree container object


        string fmContainerBase( KFMElectrostaticTreeData::Name() );
        string fmContainerName = fmContainerBase + string( "_" ) + solutionHash + string( "_" ) + parameterHash;

        if(fUseOpenCL)
        {
            fmContainerName += string("_OpenCL");
        }

        KFMElectrostaticTreeData* tree_data = new KFMElectrostaticTreeData();

        bool containerFound = false;
        KEMFileInterface::GetInstance()->FindByName( *tree_data, fmContainerName, containerFound);

        if( containerFound == true )
        {
            fieldmsg( eNormal ) << "fast multipole tree found." << eom;

            //construct tree from data
            TreeConstructor_SingleThread constructor;
            constructor.ConstructTree(*tree_data, *fTree);

        }
        else
        {
            fieldmsg_debug( "no fast multipole tree found." << eom )

            //must construct the tree
            //assign tree parameters and id
            fTree->SetParameters(fParameters);
            fTree->GetTreeProperties()->SetTreeID(fmContainerName);

            //construct the tree
            if(fUseOpenCL)
            {
                TreeConstructor_OpenCL constructor;
                #ifdef KEMFIELD_USE_OPENCL
                constructor.ConstructTree(*oclContainer, *fTree);
                #else
                constructor.ConstructTree(container, *fTree);
                #endif
            }
            else
            {
                TreeConstructor_SingleThread constructor;
                constructor.ConstructTree(container, *fTree);
            }

            TreeConstructor_SingleThread constructor;
            constructor.SaveTree(*fTree, *tree_data);

            KEMFileInterface::GetInstance()->Write( *tree_data, fmContainerName);
        }

        //now build the field solver
        if(fUseOpenCL)
        {
            #ifdef KEMFIELD_USE_OPENCL
            fFastMultipoleFieldSolverOpenCL = new KFMElectrostaticFastMultipoleFieldSolver_OpenCL(*oclContainer, *fTree);
            return;
            #endif
        }

        fFastMultipoleFieldSolver = new KFMElectrostaticFastMultipoleFieldSolver(container, *fTree);
        return;
    }

    double KSFieldElectrostatic::FastMultipoleFieldSolver::Potential( const KPosition& P ) const
    {
        if( fUseOpenCL )
        {
            #ifdef KEMFIELD_USE_OPENCL
            return fFastMultipoleFieldSolverOpenCL->Potential( P );
            #endif
        }
        return fFastMultipoleFieldSolver->Potential( P );
    }

    KEMThreeVector KSFieldElectrostatic::FastMultipoleFieldSolver::ElectricField( const KPosition& P ) const
    {
        if( fUseOpenCL )
        {
            #ifdef KEMFIELD_USE_OPENCL
            return fFastMultipoleFieldSolverOpenCL->ElectricField( P );
            #endif
        }
        return fFastMultipoleFieldSolver->ElectricField( P );
    }

////////////////////////////////////////////////////////////////////////////////

    void KSFieldElectrostatic::InitializeComponent()
    {
        if( !fBEMSolver )
        {
            fieldmsg( eError ) << "tried to initialize electrostatic field solver <" << GetName() << "> without bem solver set" << eom;
            return;
        }

        fBEMSolver->SetHashProperties( fHashMaskedBits, fHashThreshold );

        if( !fFieldSolver )
        {
            fieldmsg( eError ) << "tried to initialize electrostatic field solver <" << GetName() << "> without field solver set" << eom;
            return;
        }

        KEMFileInterface::GetInstance()->ActiveDirectory( fDirectory );
        KEMFileInterface::GetInstance()->ActiveFile( KEMFileInterface::GetInstance()->ActiveDirectory() + "/" + fFile );

        fContainer = new KSurfaceContainer();

        switch( fSymmetry )
        {
            case sNoSymmetry :
                fConverter = new KGBEMMeshConverter();
                break;

            case sAxialSymmetry :
                fConverter = new KGBEMAxialMeshConverter();
                break;

            case sDiscreteAxialSymmetry :
                fConverter = new KGBEMDiscreteRotationalMeshConverter();
                break;

            default :
                fieldmsg( eError ) << "got unknown symmetry flag <" << fSymmetry << ">" << eom;
                break;
        }

        fConverter->SetSurfaceContainer( fContainer );

        if( fSystem != NULL )
        {
            fConverter->SetSystem( fSystem->GetOrigin(), fSystem->GetXAxis(), fSystem->GetYAxis(), fSystem->GetZAxis() );
        }

        for( vector< KGSurface* >::iterator tSurfaceIt = fSurfaces.begin(); tSurfaceIt != fSurfaces.end(); tSurfaceIt++ )
        {
            (*tSurfaceIt)->AcceptNode( fConverter );
        }

        for( vector< KGSpace* >::iterator tSpaceIt = fSpaces.begin(); tSpaceIt != fSpaces.end(); tSpaceIt++ )
        {
            (*tSpaceIt)->AcceptNode( fConverter );
        }

        if( fContainer->empty() )
        {
            fieldmsg( eError ) << "electrostatic field solver <" << GetName() << "> has zero surface elements" << eom;
        }

        for (vector<Visitor*>::iterator it = fVisitors.begin();it != fVisitors.end();++it)
          if ((*it)->Preprocessing())
            (*it)->Visit(*this);

        fBEMSolver->Initialize( *fContainer );

        fFieldSolver->Initialize( *fContainer );

        for (vector<Visitor*>::iterator it = fVisitors.begin();it != fVisitors.end();++it)
          if ((*it)->Postprocessing())
            (*it)->Visit(*this);

        return;
    }

    void KSFieldElectrostatic::DeinitializeComponent()
    {

    }

}
