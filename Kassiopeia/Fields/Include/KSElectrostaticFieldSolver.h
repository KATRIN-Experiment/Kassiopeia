#ifndef Kassiopeia_KSElectrostaticFieldSolver_h_
#define Kassiopeia_KSElectrostaticFieldSolver_h_

#include "KSElectricField.h"
#include "KSFieldsMessage.h"

#include <iostream>

#include "KGBEM.hh"
#include "KGBEMConverter.hh"

#include "KTypeManipulation.hh"
using KEMField::Int2Type;
using KEMField::IsDerivedFrom;

#include "KMD5HashGenerator.hh"
#include "KEMFileInterface.hh"
using KEMField::KMD5HashGenerator;
using KEMField::Type2Type;
using KEMField::KEMFileInterface;

#include "KSurfaceContainer.hh"
#include "KElectrostaticBoundaryIntegrator.hh"
#include "KSquareMatrix.hh"
#include "KBoundaryIntegralMatrix.hh"
#include "KBoundaryIntegralVector.hh"
#include "KBoundaryIntegralSolutionVector.hh"
using KEMField::KSurfaceContainer;
using KEMField::KSurfacePrimitive;
using KEMField::KSurfaceAction;
using KEMField::KElectrostaticBoundaryIntegrator;
using KEMField::KSquareMatrix;
using KEMField::KBoundaryIntegralMatrix;
using KEMField::KBoundaryIntegralVector;
using KEMField::KBoundaryIntegralSolutionVector;
#ifdef KEMFIELD_USE_OPENCL
#include "KOpenCLSurfaceContainer.hh"
#include "KOpenCLElectrostaticBoundaryIntegrator.hh"
#include "KOpenCLBoundaryIntegralMatrix.hh"
#include "KOpenCLBoundaryIntegralVector.hh"
#include "KOpenCLBoundaryIntegralSolutionVector.hh"
using KEMField::KOpenCLInterface;
using KEMField::KOpenCLData;
using KEMField::KOpenCLSurfaceContainer;
using KEMField::KOpenCLBoundaryIntegrator;
using KEMField::KOpenCLElectrostaticBoundaryIntegrator;
#endif

#include "KGaussianElimination.hh"
using KEMField::KGaussianElimination;

#include "KSuperpositionSolver.hh"
#include "KSVDSolver.hh"
using KEMField::KSuperpositionSolver;
using KEMField::KSVDSolver;
#ifdef KEMFIELD_USE_ROOT
#include "KEMRootSVDSolver.hh"
using KEMField::KEMRootSVDSolver;
#endif

#include "KRobinHood.hh"
using KEMField::KRobinHood;

#ifdef KEMFIELD_USE_OPENCL
#include "KRobinHood_OpenCL.hh"
using KEMField::KRobinHood_OpenCL;
#endif

#ifdef KEMFIELD_USE_MPI
#include "KMPIInterface.hh"
using KEMField::KMPIInterface;
#include "KRobinHood_MPI.hh"
using KEMField::KRobinHood_MPI;
#endif

#ifdef KEMFIELD_USE_OPENCL
#ifdef KEMFIELD_USE_MPI
#include "KRobinHood_MPI_OpenCL.hh"
using KEMField::KRobinHood_MPI_OpenCL;
#endif
#endif

#include "KIterativeStateWriter.hh"
#include "KIterationTracker.hh"
using KEMField::KIterativeStateWriter;
using KEMField::KIterationTracker;
using KEMField::KResidualThreshold;
using KEMField::KIterationDisplay;
#ifdef KEMFIELD_USE_VTK
#include "KEMVTKViewer.hh"
#include "KEMVTKElectromagnetViewer.hh"
#include "KVTKIterationPlotter.hh"
using KEMField::KEMVTKViewer;
using KEMField::KEMVTKElectromagnetViewer;
using KEMField::KVTKIterationPlotter;
#endif

#include "KElectrostaticIntegratingFieldSolver.hh"
using KEMField::KElectrostaticBoundaryIntegrator;
using KEMField::KIntegratingFieldSolver;
#ifdef KEMFIELD_USE_OPENCL
#include "KOpenCLElectrostaticIntegratingFieldSolver.hh"
#endif

#include "KElectrostaticZonalHarmonicFieldSolver.hh"
#include "KZonalHarmonicContainer.hh"
#include "KZonalHarmonicParameters.hh"
using KEMField::KZonalHarmonicFieldSolver;
using KEMField::KZonalHarmonicContainer;
using KEMField::KZonalHarmonicParameters;


#include "KFMBoundaryIntegralMatrix.hh"
#include "KFMElectrostaticBoundaryIntegrator.hh"
#include "KFMElectrostaticBoundaryIntegratorEngine_SingleThread.hh"
#include "KFMElectrostaticFieldMapper_SingleThread.hh"
using KEMField::KFMBoundaryIntegralMatrix;
using KEMField::KFMElectrostaticBoundaryIntegrator;
using KEMField::KFMElectrostaticBoundaryIntegratorEngine_SingleThread;
using KEMField::KFMElectrostaticFieldMapper_SingleThread;

#include "KIterativeKrylovSolver.hh"
#include "KIterativeKrylovRestartCondition.hh"
#include "KGeneralizedMinimalResidual.hh"
#include "KBiconjugateGradientStabilized.hh"
using KEMField::KIterativeKrylovSolver;
using KEMField::KIterativeKrylovRestartCondition;
using KEMField::KGeneralizedMinimalResidual;
using KEMField::KBiconjugateGradientStabilized;

#include "KFMElectrostaticFastMultipoleFieldSolver.hh"
#include "KFMElectrostaticTree.hh"
#include "KFMElectrostaticTreeData.hh"
#include "KFMElectrostaticTreeConstructor.hh"

using KEMField::KFMElectrostaticFastMultipoleFieldSolver;
using KEMField::KFMElectrostaticTree;
using KEMField::KFMElectrostaticTreeData;
using KEMField::KFMElectrostaticTreeConstructor;

#include "KSAStructuredASCIIHeaders.hh"
using KEMField::KSAObjectInputNode;
using KEMField::KSAObjectOutputNode;


#ifdef KEMFIELD_USE_OPENCL
#include "KFMElectrostaticFieldMapper_OpenCL.hh"
#include "KFMElectrostaticBoundaryIntegratorEngine_OpenCL.hh"
#include "KFMElectrostaticFastMultipoleFieldSolver_OpenCL.hh"
using KEMField::KFMElectrostaticFieldMapper_OpenCL;
using KEMField::KFMElectrostaticBoundaryIntegratorEngine_OpenCL;
using KEMField::KFMElectrostaticFastMultipoleFieldSolver_OpenCL;
#endif


#ifdef KEMFIELD_USE_MPI
#define MPI_SINGLE_PROCESS if ( KEMField::KMPIInterface::GetInstance()->GetProcess()==0 )
#else
#define MPI_SINGLE_PROCESS if( true )
#endif

using namespace KGeoBag;

namespace Kassiopeia
{

    class KSElectrostaticFieldSolver :
        public KSComponent< KSElectrostaticFieldSolver, KSElectricField >
    {
        public:
            KSElectrostaticFieldSolver();
            KSElectrostaticFieldSolver( const KSElectrostaticFieldSolver& aCopy );
            KSElectrostaticFieldSolver* Clone() const;
            virtual ~KSElectrostaticFieldSolver();

        public:
            void CalculatePotential( const KThreeVector& aSamplePoint, const double& aSampleTime, double& aPotential );
            void CalculateField( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeVector& aField );

        public:
            static const unsigned int sNoSymmetry;
            static const unsigned int sAxialSymmetry;
            static const unsigned int sDiscreteAxialSymmetry;

            void SetDirectory( const string& aDirectory );
            void SetFile( const string& aFile );

            void SetHashMaskedBits( const unsigned int& aMaskedBits );
            void SetHashThreshold( const double& aThreshold );

            void SetSystem( KGSpace* aSpace );
            void AddSurface( KGSurface* aSurface );
            void AddSpace( KGSpace* aSpace );
            void SetSymmetry( const unsigned int& aSymmetry );

        private:
            string fDirectory;
            unsigned int fHashMaskedBits;
            double fHashThreshold;
            string fFile;
            KGSpace* fSystem;
            vector< KGSurface* > fSurfaces;
            vector< KGSpace* > fSpaces;
            unsigned int fSymmetry;

        public:
            class Visitor
	    {
	    public:
	      Visitor();
	      virtual ~Visitor() {}

	      void Preprocessing(bool choice)
	      {
		fPreprocessing = choice;
	      }
	      void Postprocessing(bool choice)
	      {
		fPostprocessing = choice;
	      }

	      bool Preprocessing() const
	      {
		return fPreprocessing;
	      }
	      bool Postprocessing() const
	      {
		return fPostprocessing;
	      }

	      virtual void Visit(KSElectrostaticFieldSolver&) = 0;

	    protected:
	      bool fPreprocessing;
	      bool fPostprocessing;
	    };

            class VTKViewer : public KSElectrostaticFieldSolver::Visitor
	    {
	    public:
	      VTKViewer();
	      virtual ~VTKViewer();

	      void ViewGeometry(bool choice)
	      {
		fViewGeometry = choice;
	      }
	      void SaveGeometry(bool choice)
	      {
		fSaveGeometry = choice;
	      }
	      void SetFile(string file)
	      {
		fFile = file;
	      }

	      bool ViewGeometry() const
	      {
		return fViewGeometry;
	      }
	      bool SaveGeometry() const
	      {
		return fSaveGeometry;
	      }

	      void Visit(KSElectrostaticFieldSolver&);

	    protected:

	      bool fViewGeometry;
	      bool fSaveGeometry;
	      std::string fFile;
	    };

            class BEMSolver
            {
                public:
                    BEMSolver();
                    virtual ~BEMSolver();

                    virtual void Initialize( KSurfaceContainer& container ) = 0;

                    void SetHashProperties( unsigned int maskedBits, double hashThreshold );

                protected:
                    bool FindSolution( double threshold, KSurfaceContainer& container );
                    void SaveSolution( double threshold, KSurfaceContainer& container );

                    unsigned int fHashMaskedBits;
                    double fHashThreshold;
            };

            class CachedBEMSolver :
                public BEMSolver
            {
                public:
                    CachedBEMSolver();
                    virtual ~CachedBEMSolver();

                    void Initialize( KSurfaceContainer& container );

                    void SetName( std::string s )
                    {
                        fName = s;
                    }
                    void SetHash( std::string s )
                    {
                        fHash = s;
                    }

                private:
                    std::string fName;
                    std::string fHash;
            };

            class GaussianEliminationBEMSolver :
                public BEMSolver
            {
                public:
                    GaussianEliminationBEMSolver();
                    virtual ~GaussianEliminationBEMSolver();

                    void Initialize( KSurfaceContainer& container );

            };

            class RobinHoodBEMSolver :
                public BEMSolver
            {
                public:
                    RobinHoodBEMSolver();
                    virtual ~RobinHoodBEMSolver();

                    void Initialize( KSurfaceContainer& container );

                    void SetTolerance( double d )
                    {
                        fTolerance = d;
                    }
                    void SetCheckSubInterval( unsigned int i )
                    {
                        fCheckSubInterval = i;
                    }
                    void SetDisplayInterval( unsigned int i )
                    {
                        fDisplayInterval = i;
                    }
                    void SetWriteInterval( unsigned int i )
                    {
                        fWriteInterval = i;
                    }
                    void SetPlotInterval( unsigned int i )
                    {
                        fPlotInterval = i;
                    }

                    void CacheMatrixElements( bool choice )
                    {
                        fCacheMatrixElements = choice;
                    }

                    void UseOpenCL( bool choice )
                    {
                        if( choice == true )
                        {
#ifdef KEMFIELD_USE_OPENCL
                            fUseOpenCL = choice;
                            return;
#endif
                            fieldmsg( eWarning ) << "cannot use opencl in robin hood without kemfield being built with opencl, using defaults." << eom;
                        }
                        fUseOpenCL = false;
                        return;
                    }
                    void UseVTK( bool choice )
                    {
                        if( choice == true )
                        {
#ifdef KEMFIELD_USE_VTK
                            fUseVTK = choice;
                            return;
#endif
                            fieldmsg( eWarning ) << "cannot use vtk in robin hood without kemfield being built with vtk, using defaults." << eom;
                        }
                        fUseVTK = false;
                        return;
                    }

                private:
                    double fTolerance;
                    unsigned int fCheckSubInterval;
                    unsigned int fDisplayInterval;
                    unsigned int fWriteInterval;
                    unsigned int fPlotInterval;
                    bool fCacheMatrixElements;
                    bool fUseOpenCL;
                    bool fUseVTK;
            };


            class FastMultipoleBEMSolver :
                public BEMSolver
            {
                public:
                    FastMultipoleBEMSolver();
                    virtual ~FastMultipoleBEMSolver();

                    void Initialize( KSurfaceContainer& container );

                    void SetTolerance( double d )
                    {
                        fTolerance = d;
                    }

                    void SetKrylovSolverType( std::string krylov )
                    {
                        fKrylov = krylov;
                    }

                    void SetRestartCycleSize( unsigned int i)
                    {
                        fRestartCycle = i;
                    }

                    void UseOpenCL( bool choice )
                    {
                        if( choice == true )
                        {
#ifdef KEMFIELD_USE_OPENCL
                            fUseOpenCL = choice;
                            return;
#endif
                            fieldmsg( eWarning ) << "cannot use opencl in fast multipole without kemfield being built with opencl, using defaults." << eom;
                        }
                        fUseOpenCL = false;
                        return;
                    }
                    void UseVTK( bool choice )
                    {
                        if( choice == true )
                        {
#ifdef KEMFIELD_USE_VTK
                            fUseVTK = choice;
                            return;
#endif
                            fieldmsg( eWarning ) << "cannot use vtk in fast multipole without kemfield being built with vtk, using defaults." << eom;
                        }
                        fUseVTK = false;
                        return;
                    }

                    KFMElectrostaticParameters* GetParameters(){ return &fParameters; };

                private:

                    double fTolerance;
                    KFMElectrostaticParameters fParameters;
                    std::string fKrylov;
                    unsigned int fRestartCycle;
                    bool fUseOpenCL;
                    bool fUseVTK;
            };


            class FieldSolver
            {
                public:
                    FieldSolver();
                    virtual ~FieldSolver();

                    virtual void Initialize( KSurfaceContainer& container ) = 0;

                    virtual double Potential( const KPosition& P ) const = 0;
                    virtual KEMThreeVector ElectricField( const KPosition& P ) const = 0;

            };

            class IntegratingFieldSolver :
                public FieldSolver
            {
                public:
                    IntegratingFieldSolver();
                    virtual ~IntegratingFieldSolver();

                    void Initialize( KSurfaceContainer& container );

                    double Potential( const KPosition& P ) const;
                    KEMThreeVector ElectricField( const KPosition& P ) const;

                    void UseOpenCL( bool choice )
                    {
                        fUseOpenCL = choice;
                    }

                private:
                    KElectrostaticBoundaryIntegrator* fIntegrator;
                    KIntegratingFieldSolver< KElectrostaticBoundaryIntegrator >* fIntegratingFieldSolver;

#ifdef KEMFIELD_USE_OPENCL
                    KOpenCLElectrostaticBoundaryIntegrator* fOCLIntegrator;
                    KIntegratingFieldSolver< KOpenCLElectrostaticBoundaryIntegrator >* fOCLIntegratingFieldSolver;
#endif

                    bool fUseOpenCL;
            };

            class ZonalHarmonicFieldSolver :
                public FieldSolver
            {
                public:
                    ZonalHarmonicFieldSolver();
                    virtual ~ZonalHarmonicFieldSolver();

                    void Initialize( KSurfaceContainer& container );

                    double Potential( const KPosition& P ) const;
                    KEMThreeVector ElectricField( const KPosition& P ) const;

                    KZonalHarmonicParameters* GetParameters()
                    {
                        return fParameters;
                    }

                private:
                    KElectrostaticBoundaryIntegrator fIntegrator;
                    KZonalHarmonicContainer< KElectrostaticBasis >* fZHContainer;
                    KZonalHarmonicFieldSolver< KElectrostaticBasis >* fZonalHarmonicFieldSolver;
                    KZonalHarmonicParameters* fParameters;
            };


            class FastMultipoleFieldSolver :
                public FieldSolver
            {
                public:
                    FastMultipoleFieldSolver();
                    virtual ~FastMultipoleFieldSolver();

                    void Initialize( KSurfaceContainer& container );

                    double Potential( const KPosition& P ) const;
                    KEMThreeVector ElectricField( const KPosition& P ) const;

                    KFMElectrostaticParameters* GetParameters(){ return &fParameters; };

                    void UseOpenCL( bool choice )
                    {
                        if( choice == true )
                        {
#ifdef KEMFIELD_USE_OPENCL
                            fUseOpenCL = choice;
                            return;
#endif
                            fieldmsg( eWarning ) << "cannot use opencl in fast multipole without kemfield being built with opencl, using defaults." << eom;
                        }
                        fUseOpenCL = false;
                        return;
                    }

                private:

                    KFMElectrostaticParameters fParameters;
                    KFMElectrostaticTree* fTree;
                    KFMElectrostaticFastMultipoleFieldSolver* fFastMultipoleFieldSolver;
                    #ifdef KEMFIELD_USE_OPENCL
                    KFMElectrostaticFastMultipoleFieldSolver_OpenCL* fFastMultipoleFieldSolverOpenCL;
                    #endif
                    bool fUseOpenCL;
            };




        public:
            KGBEMConverter* GetConverter()
            {
                return fConverter;
            }
            KSurfaceContainer* GetContainer()
            {
                return fContainer;
            }

            void AddVisitor( Visitor* visitor )
            {
	        fVisitors.push_back(visitor);
                return;
            }
            vector<Visitor*>& GetVisitors()
            {
	         return fVisitors;
            }

            void SetBEMSolver( BEMSolver* solver )
            {
                if( fBEMSolver != NULL )
                {
                    fieldmsg( eError ) << "tried to assign more than one electrostatic bem solver" << eom;
                    return;
                }
                fBEMSolver = solver;
                return;
            }
            BEMSolver* GetBEMSolver()
            {
                return fBEMSolver;
            }

            void SetFieldSolver( FieldSolver* solver )
            {
                if( fFieldSolver != NULL )
                {
                    fieldmsg( eError ) << "tried to assign more than one electrostatic field solver" << eom;
                    return;
                }
                fFieldSolver = solver;
                return;
            }
            FieldSolver* GetFieldSolver()
            {
                return fFieldSolver;
            }

        private:
            void InitializeComponent();
            void DeinitializeComponent();

            KGBEMConverter* fConverter;
            KSurfaceContainer* fContainer;
            vector<Visitor*> fVisitors;
            BEMSolver* fBEMSolver;
            FieldSolver* fFieldSolver;
    };
}

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////                                                   /////
/////  BBBB   U   U  IIIII  L      DDDD   EEEEE  RRRR   /////
/////  B   B  U   U    I    L      D   D  E      R   R  /////
/////  BBBB   U   U    I    L      D   D  EE     RRRR   /////
/////  B   B  U   U    I    L      D   D  E      R   R  /////
/////  BBBB    UUU   IIIII  LLLLL  DDDD   EEEEE  R   R  /////
/////                                                   /////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

#include "KComplexElement.hh"
#include "KSFieldsMessage.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSElectrostaticFieldSolver::VTKViewer > KSKEMFieldVTKViewerBuilder;

    template< >
    inline bool KSKEMFieldVTKViewerBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "file" )
        {
            std::string name;
            aContainer->CopyTo( name );
            fObject->SetFile( name );
            return true;
        }
        if( aContainer->GetName() == "view" )
        {
            bool choice;
            aContainer->CopyTo( choice );
            fObject->ViewGeometry( choice );
            return true;
        }
        if( aContainer->GetName() == "save" )
        {
            bool choice;
            aContainer->CopyTo( choice );
            fObject->SaveGeometry( choice );
            return true;
        }
        if( aContainer->GetName() == "preprocessing" )
        {
            bool choice;
            aContainer->CopyTo( choice );
            fObject->Preprocessing( choice );
            return true;
        }
        if( aContainer->GetName() == "postprocessing" )
        {
            bool choice;
            aContainer->CopyTo( choice );
            fObject->Postprocessing( choice );
            return true;
        }
        return false;
    }

    typedef KComplexElement< KSElectrostaticFieldSolver::CachedBEMSolver > KSCachedBEMSolverBuilder;

    template< >
    inline bool KSCachedBEMSolverBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            std::string name;
            aContainer->CopyTo( name );
            fObject->SetName( name );
            return true;
        }
        if( aContainer->GetName() == "hash" )
        {
            std::string hash;
            aContainer->CopyTo( hash );
            fObject->SetHash( hash );
            return true;
        }
        return false;
    }

    typedef KComplexElement< KSElectrostaticFieldSolver::RobinHoodBEMSolver > KSRobinHoodBEMSolverBuilder;

    template< >
    inline bool KSRobinHoodBEMSolverBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "tolerance" )
        {
            aContainer->CopyTo( fObject, &KSElectrostaticFieldSolver::RobinHoodBEMSolver::SetTolerance );
            return true;
        }
        if( aContainer->GetName() == "check_sub_interval" )
        {
            aContainer->CopyTo( fObject, &KSElectrostaticFieldSolver::RobinHoodBEMSolver::SetCheckSubInterval );
            return true;
        }
        if( aContainer->GetName() == "display_interval" )
        {
            aContainer->CopyTo( fObject, &KSElectrostaticFieldSolver::RobinHoodBEMSolver::SetDisplayInterval );
            return true;
        }
        if( aContainer->GetName() == "write_interval" )
        {
            aContainer->CopyTo( fObject, &KSElectrostaticFieldSolver::RobinHoodBEMSolver::SetWriteInterval );
            return true;
        }
        if( aContainer->GetName() == "plot_interval" )
        {
            aContainer->CopyTo( fObject, &KSElectrostaticFieldSolver::RobinHoodBEMSolver::SetPlotInterval );
            return true;
        }
        if( aContainer->GetName() == "cache_matrix_elements" )
        {
            aContainer->CopyTo( fObject, &KSElectrostaticFieldSolver::RobinHoodBEMSolver::CacheMatrixElements );
            return true;
        }
        if( aContainer->GetName() == "use_opencl" )
        {
            aContainer->CopyTo( fObject, &KSElectrostaticFieldSolver::RobinHoodBEMSolver::UseOpenCL );
            return true;
        }
        if( aContainer->GetName() == "use_vtk" )
        {
            aContainer->CopyTo( fObject, &KSElectrostaticFieldSolver::RobinHoodBEMSolver::UseVTK );
            return true;
        }
        return false;
    }


    typedef KComplexElement< KSElectrostaticFieldSolver::FastMultipoleBEMSolver > KSFastMultipoleBEMSolverBuilder;

    template< >
    inline bool KSFastMultipoleBEMSolverBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "tolerance" )
        {
            aContainer->CopyTo( fObject, &KSElectrostaticFieldSolver::FastMultipoleBEMSolver::SetTolerance );
            return true;
        }
        if( aContainer->GetName() == "krylov_solver_type" )
        {
            aContainer->CopyTo( fObject, &KSElectrostaticFieldSolver::FastMultipoleBEMSolver::SetKrylovSolverType );
            return true;
        }
        if( aContainer->GetName() == "restart_cycle_size" )
        {
            aContainer->CopyTo( fObject, &KSElectrostaticFieldSolver::FastMultipoleBEMSolver::SetRestartCycleSize );
            return true;
        }
        if( aContainer->GetName() == "spatial_division" )
        {
            unsigned int nDivisions;
            aContainer->CopyTo( nDivisions );
            fObject->GetParameters()->divisions = nDivisions;
            return true;
        }
        if( aContainer->GetName() == "expansion_degree" )
        {
            unsigned int nDegree;
            aContainer->CopyTo( nDegree );
            fObject->GetParameters()->degree = nDegree;
            return true;
        }
        if( aContainer->GetName() == "neighbor_order" )
        {
            unsigned int nNeighborOrder;
            aContainer->CopyTo( nNeighborOrder );
            fObject->GetParameters()->zeromask = nNeighborOrder;
            return true;
        }
        if( aContainer->GetName() == "maximum_tree_depth" )
        {
            unsigned int nMaxTreeDepth;
            aContainer->CopyTo( nMaxTreeDepth );
            fObject->GetParameters()->maximum_tree_depth = nMaxTreeDepth;
            return true;
        }
        if( aContainer->GetName() == "region_expansion_factor" )
        {
            double dExpansionFactor;
            aContainer->CopyTo( dExpansionFactor );
            fObject->GetParameters()->region_expansion_factor = dExpansionFactor;
            return true;
        }
        if( aContainer->GetName() == "use_region_size_estimation" )
        {
            bool useEstimation;
            aContainer->CopyTo( useEstimation );
            fObject->GetParameters()->use_region_estimation = useEstimation;
            return true;
        }
        if( aContainer->GetName() == "world_cube_center_x" )
        {
            double x;
            aContainer->CopyTo( x );
            fObject->GetParameters()->world_center_x = x;
            return true;
        }
        if( aContainer->GetName() == "world_cube_center_y" )
        {
            double y;
            aContainer->CopyTo( y );
            fObject->GetParameters()->world_center_y = y;
            return true;
        }
        if( aContainer->GetName() == "world_cube_center_z" )
        {
            double z;
            aContainer->CopyTo( z );
            fObject->GetParameters()->world_center_z = z;
            return true;
        }
        if( aContainer->GetName() == "world_cube_length" )
        {
            double l;
            aContainer->CopyTo( l );
            fObject->GetParameters()->world_length = l;
            return true;
        }
        if( aContainer->GetName() == "use_caching" )
        {
            bool b;
            aContainer->CopyTo( b );
            fObject->GetParameters()->use_caching = b;
            return true;
        }
        if( aContainer->GetName() == "verbosity" )
        {
            unsigned int n;
            aContainer->CopyTo( n );
            fObject->GetParameters()->verbosity = n;
            return true;
        }
        if( aContainer->GetName() == "use_opencl" )
        {
            aContainer->CopyTo( fObject, &KSElectrostaticFieldSolver::FastMultipoleBEMSolver::UseOpenCL );
            return true;
        }
        if( aContainer->GetName() == "use_vtk" )
        {
            aContainer->CopyTo( fObject, &KSElectrostaticFieldSolver::FastMultipoleBEMSolver::UseVTK );
            return true;
        }
        return false;
    }

    typedef KComplexElement< KSElectrostaticFieldSolver::GaussianEliminationBEMSolver > KSGaussianEliminationBEMSolverBuilder;

    typedef KComplexElement< KSElectrostaticFieldSolver::IntegratingFieldSolver > KSElectrostaticIntegratingFieldSolverBuilder;

    template< >
    inline bool KSElectrostaticIntegratingFieldSolverBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "use_opencl" )
        {
            aContainer->CopyTo( fObject, &KSElectrostaticFieldSolver::IntegratingFieldSolver::UseOpenCL );
            return true;
        }
        return false;
    }

    typedef KComplexElement< KSElectrostaticFieldSolver::ZonalHarmonicFieldSolver > KSElectrostaticZonalHarmonicFieldSolverBuilder;

    template< >
    inline bool KSElectrostaticZonalHarmonicFieldSolverBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "number_of_bifurcations" )
        {
            int nBifurcations;
            aContainer->CopyTo( nBifurcations );
            fObject->GetParameters()->SetNBifurcations( nBifurcations );
            return true;
        }
        if( aContainer->GetName() == "convergence_ratio" )
        {
            double convergenceRatio;
            aContainer->CopyTo( convergenceRatio );
            fObject->GetParameters()->SetConvergenceRatio( convergenceRatio );
            return true;
        }
        if( aContainer->GetName() == "proximity_to_sourcepoint" )
        {
            double proximityToSourcePoint;
            aContainer->CopyTo( proximityToSourcePoint );
            fObject->GetParameters()->SetProximityToSourcePoint( proximityToSourcePoint );
            return true;
        }
        if( aContainer->GetName() == "convergence_parameter" )
        {
            double convergenceParameter;
            aContainer->CopyTo( convergenceParameter );
            fObject->GetParameters()->SetConvergenceParameter( convergenceParameter );
            return true;
        }
        if( aContainer->GetName() == "number_of_central_coefficients" )
        {
            int nCentralCoefficients;
            aContainer->CopyTo( nCentralCoefficients );
            fObject->GetParameters()->SetNCentralCoefficients( nCentralCoefficients );
            return true;
        }
        if( aContainer->GetName() == "use_fractional_central_sourcepoint_spacing" )
        {
            bool centralFractionalSpacing;
            aContainer->CopyTo( centralFractionalSpacing );
            fObject->GetParameters()->SetCentralFractionalSpacing( centralFractionalSpacing );
            return true;
        }
        if( aContainer->GetName() == "central_sourcepoint_fractional_distance" )
        {
            double centralFractionalDistance;
            aContainer->CopyTo( centralFractionalDistance );
            fObject->GetParameters()->SetCentralFractionalDistance( centralFractionalDistance );
            return true;
        }
        if( aContainer->GetName() == "central_sourcepoint_spacing" )
        {
            double centralDeltaZ;
            aContainer->CopyTo( centralDeltaZ );
            fObject->GetParameters()->SetCentralDeltaZ( centralDeltaZ );
            return true;
        }
        if( aContainer->GetName() == "central_sourcepoint_start" )
        {
            double centralZ1;
            aContainer->CopyTo( centralZ1 );
            fObject->GetParameters()->SetCentralZ1( centralZ1 );
            return true;
        }
        if( aContainer->GetName() == "central_sourcepoint_end" )
        {
            double centralZ2;
            aContainer->CopyTo( centralZ2 );
            fObject->GetParameters()->SetCentralZ2( centralZ2 );
            return true;
        }
        if( aContainer->GetName() == "number_of_remote_coefficients" )
        {
            int nRemoteCoefficients;
            aContainer->CopyTo( nRemoteCoefficients );
            fObject->GetParameters()->SetNRemoteCoefficients( nRemoteCoefficients );
            return true;
        }
        if( aContainer->GetName() == "remote_sourcepoint_start" )
        {
            double remoteZ1;
            aContainer->CopyTo( remoteZ1 );
            fObject->GetParameters()->SetRemoteZ1( remoteZ1 );
            return true;
        }
        if( aContainer->GetName() == "remote_sourcepoint_end" )
        {
            double remoteZ2;
            aContainer->CopyTo( remoteZ2 );
            fObject->GetParameters()->SetRemoteZ2( remoteZ2 );
            return true;
        }
        return false;
    }



    typedef KComplexElement< KSElectrostaticFieldSolver::FastMultipoleFieldSolver > KSFastMultipoleFieldSolverBuilder;

    template< >
    inline bool KSFastMultipoleFieldSolverBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "spatial_division" )
        {
            unsigned int nDivisions;
            aContainer->CopyTo( nDivisions );
            fObject->GetParameters()->divisions = nDivisions;
            return true;
        }
        if( aContainer->GetName() == "expansion_degree" )
        {
            unsigned int nDegree;
            aContainer->CopyTo( nDegree );
            fObject->GetParameters()->degree = nDegree;
            return true;
        }
        if( aContainer->GetName() == "neighbor_order" )
        {
            unsigned int nNeighborOrder;
            aContainer->CopyTo( nNeighborOrder );
            fObject->GetParameters()->zeromask = nNeighborOrder;
            return true;
        }
        if( aContainer->GetName() == "maximum_tree_depth" )
        {
            unsigned int nMaxTreeDepth;
            aContainer->CopyTo( nMaxTreeDepth );
            fObject->GetParameters()->maximum_tree_depth = nMaxTreeDepth;
            return true;
        }
        if( aContainer->GetName() == "region_expansion_factor" )
        {
            double dExpansionFactor;
            aContainer->CopyTo( dExpansionFactor );
            fObject->GetParameters()->region_expansion_factor = dExpansionFactor;
            return true;
        }
        if( aContainer->GetName() == "use_region_size_estimation" )
        {
            bool useEstimation;
            aContainer->CopyTo( useEstimation );
            fObject->GetParameters()->use_region_estimation = useEstimation;
            return true;
        }
        if( aContainer->GetName() == "world_cube_center_x" )
        {
            double x;
            aContainer->CopyTo( x );
            fObject->GetParameters()->world_center_x = x;
            return true;
        }
        if( aContainer->GetName() == "world_cube_center_y" )
        {
            double y;
            aContainer->CopyTo( y );
            fObject->GetParameters()->world_center_y = y;
            return true;
        }
        if( aContainer->GetName() == "world_cube_center_z" )
        {
            double z;
            aContainer->CopyTo( z );
            fObject->GetParameters()->world_center_z = z;
            return true;
        }
        if( aContainer->GetName() == "world_cube_length" )
        {
            double l;
            aContainer->CopyTo( l );
            fObject->GetParameters()->world_length = l;
            return true;

        }
        if( aContainer->GetName() == "use_caching" )
        {
            bool b;
            aContainer->CopyTo( b );
            fObject->GetParameters()->use_caching = b;
            return true;
        }
        if( aContainer->GetName() == "verbosity" )
        {
            unsigned int n;
            aContainer->CopyTo( n );
            fObject->GetParameters()->verbosity = n;
            return true;
        }
        if( aContainer->GetName() == "use_opencl" )
        {
            bool choice;
            aContainer->CopyTo(choice);
            fObject->UseOpenCL(choice);
            return true;
        }
        return false;
    }



    typedef KComplexElement< KSElectrostaticFieldSolver > KSElectrostaticFieldSolverBuilder;

    template< >
    inline bool KSElectrostaticFieldSolverBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KSElectrostaticFieldSolver::SetName );
            return true;
        }
        if( aContainer->GetName() == "directory" )
        {
            aContainer->CopyTo( fObject, &KSElectrostaticFieldSolver::SetDirectory );
            return true;
        }
        if( aContainer->GetName() == "file" )
        {
            aContainer->CopyTo( fObject, &KSElectrostaticFieldSolver::SetFile );
            return true;
        }
        if( aContainer->GetName() == "system" )
        {
            KGSpace* tSpace = KGInterface::GetInstance()->RetrieveSpace( aContainer->AsReference< string >() );

            if( tSpace == NULL )
            {
                coremsg( eWarning ) << "no spaces found for specifier <" << aContainer->AsReference< string >() << ">" << eom;
                return false;
            }

            fObject->SetSystem( tSpace );

            return true;
        }
        if( aContainer->GetName() == "surfaces" )
        {
            vector< KGSurface* > tSurfaces = KGInterface::GetInstance()->RetrieveSurfaces( aContainer->AsReference< string >() );
            vector< KGSurface* >::const_iterator tSurfaceIt;
            KGSurface* tSurface;

            if( tSurfaces.size() == 0 )
            {
                coremsg( eWarning ) << "no surfaces found for specifier <" << aContainer->AsReference< string >() << ">" << eom;
                return false;
            }

            for( tSurfaceIt = tSurfaces.begin(); tSurfaceIt != tSurfaces.end(); tSurfaceIt++ )
            {
                tSurface = *tSurfaceIt;
                fObject->AddSurface( tSurface );
            }
            return true;
        }
        if( aContainer->GetName() == "spaces" )
        {
            vector< KGSpace* > tSpaces = KGInterface::GetInstance()->RetrieveSpaces( aContainer->AsReference< string >() );
            vector< KGSpace* >::const_iterator tSpaceIt;
            KGSpace* tSpace;

            if( tSpaces.size() == 0 )
            {
                coremsg( eWarning ) << "no spaces found for specifier <" << aContainer->AsReference< string >() << ">" << eom;
                return false;
            }

            for( tSpaceIt = tSpaces.begin(); tSpaceIt != tSpaces.end(); tSpaceIt++ )
            {
                tSpace = *tSpaceIt;
                fObject->AddSpace( tSpace );
            }
            return true;
        }
        if( aContainer->GetName() == "symmetry" )
        {
            if( aContainer->AsReference< string >() == "none" )
            {
                fObject->SetSymmetry( KSElectrostaticFieldSolver::sNoSymmetry );
                return true;
            }
            if( aContainer->AsReference< string >() == "axial" )
            {
                fObject->SetSymmetry( KSElectrostaticFieldSolver::sAxialSymmetry );
                return true;
            }
            if( aContainer->AsReference< string >() == "discrete_axial" )
            {
                fObject->SetSymmetry( KSElectrostaticFieldSolver::sDiscreteAxialSymmetry );
                return true;
            }
            fieldmsg( eWarning ) << "symmetry must be <none>, <axial>, or <discrete_axial>" << eom;
            return false;
        }
        if( aContainer->GetName() == "hash_masked_bits" )
        {
            aContainer->CopyTo( fObject, &KSElectrostaticFieldSolver::SetHashMaskedBits );
            return true;
        }
        if( aContainer->GetName() == "hash_threshold" )
        {
            aContainer->CopyTo( fObject, &KSElectrostaticFieldSolver::SetHashThreshold );
            return true;
        }
        return false;
    }

    typedef KComplexElement< KSElectrostaticFieldSolver > KSElectrostaticFieldSolverBuilder;

    template< >
    inline bool KSElectrostaticFieldSolverBuilder::AddElement( KContainer* anElement )
    {
        if( anElement->GetName() == "viewer" )
        {
            anElement->ReleaseTo( fObject, &KSElectrostaticFieldSolver::AddVisitor );
            return true;
        }
        if( anElement->GetName() == "cached_bem_solver" )
        {
            anElement->ReleaseTo( fObject, &KSElectrostaticFieldSolver::SetBEMSolver );
            return true;
        }
        if( anElement->GetName() == "gaussian_elimination_bem_solver" )
        {
            anElement->ReleaseTo( fObject, &KSElectrostaticFieldSolver::SetBEMSolver );
            return true;
        }
        if( anElement->GetName() == "robin_hood_bem_solver" )
        {
            anElement->ReleaseTo( fObject, &KSElectrostaticFieldSolver::SetBEMSolver );
            return true;
        }
        if( anElement->GetName() == "fast_multipole_bem_solver" )
        {
            anElement->ReleaseTo( fObject, &KSElectrostaticFieldSolver::SetBEMSolver );
            return true;
        }
        if( anElement->GetName() == "integrating_field_solver" )
        {
            anElement->ReleaseTo( fObject, &KSElectrostaticFieldSolver::SetFieldSolver );
            return true;
        }
        if( anElement->GetName() == "zonal_harmonic_field_solver" )
        {
            anElement->ReleaseTo( fObject, &KSElectrostaticFieldSolver::SetFieldSolver );
            return true;
        }
        if( anElement->GetName() == "fast_multipole_field_solver" )
        {
            anElement->ReleaseTo( fObject, &KSElectrostaticFieldSolver::SetFieldSolver );
            return true;
        }
        return false;
    }

}

#endif
