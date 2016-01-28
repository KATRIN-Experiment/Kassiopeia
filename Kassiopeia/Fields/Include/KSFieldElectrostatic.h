#ifndef Kassiopeia_KSFieldElectrostatic_h_
#define Kassiopeia_KSFieldElectrostatic_h_

#include "KSElectricField.h"
#include "KSFieldsMessage.h"

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
#include "KProjectionSolver.hh"
#include "KSVDSolver.hh"
using KEMField::KSuperpositionSolver;
using KEMField::KProjectionSolver;
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

#include "KFMElectrostaticTree.hh"
#include "KFMElectrostaticParameters.hh"
#include "KFMElectrostaticTreeData.hh"
#include "KFMElectrostaticTreeConstructor.hh"
#include "KFMElectrostaticFastMultipoleFieldSolver.hh"
#include "KFMElectrostaticFieldMapper_SingleThread.hh"
#include "KFMElectrostaticParametersConfiguration.hh"
#include "KFMElectrostaticFastMultipoleBoundaryValueSolver.hh"
#include "KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration.hh"
using KEMField::KFMElectrostaticTree;
using KEMField::KFMElectrostaticParameters;
using KEMField::KFMElectrostaticTreeData;
using KEMField::KFMElectrostaticTreeConstructor;
using KEMField::KFMElectrostaticFastMultipoleFieldSolver;
using KEMField::KFMElectrostaticFieldMapper_SingleThread;
using KEMField::KFMElectrostaticParametersConfiguration;
using KEMField::KFMElectrostaticFastMultipoleBoundaryValueSolver;
using KEMField::KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration;

#ifdef KEMFIELD_USE_OPENCL
#include "KFMElectrostaticFieldMapper_OpenCL.hh"
#include "KFMElectrostaticFastMultipoleFieldSolver_OpenCL.hh"
using KEMField::KFMElectrostaticFieldMapper_OpenCL;
using KEMField::KFMElectrostaticFastMultipoleFieldSolver_OpenCL;
#endif

#include "KSAStructuredASCIIHeaders.hh"
using KEMField::KSAObjectInputNode;
using KEMField::KSAObjectOutputNode;

#ifdef KEMFIELD_USE_MPI
    #ifndef MPI_SINGLE_PROCESS
        #define MPI_SINGLE_PROCESS if ( KEMField::KMPIInterface::GetInstance()->GetProcess()==0 )
    #endif
#else
    #ifndef MPI_SINGLE_PROCESS
        #define MPI_SINGLE_PROCESS if( true )
    #endif
#endif

using namespace KGeoBag;

namespace Kassiopeia
{

    class KSFieldElectrostatic :
        public KSComponentTemplate< KSFieldElectrostatic, KSElectricField >
    {
        public:
            KSFieldElectrostatic();
            KSFieldElectrostatic( const KSFieldElectrostatic& aCopy );
            KSFieldElectrostatic* Clone() const;
            virtual ~KSFieldElectrostatic();

        public:
            void CalculatePotential( const KThreeVector& aSamplePoint, const double& aSampleTime, double& aPotential );
            void CalculateField( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeVector& aField );
            void CalculateFieldAndPotential( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeVector& aField, double& aPotential );

        public:
            static const unsigned int sNoSymmetry;
            static const unsigned int sAxialSymmetry;
            static const unsigned int sDiscreteAxialSymmetry;

            void SetDirectory( const string& aDirectory );
            void SetFile( const string& aFile );

            void SetHashMaskedBits( const unsigned int& aMaskedBits );
            void SetHashThreshold( const double& aThreshold );

            void SetMinimumElementArea( const double& aArea);
            void SetMaximumElementAspectRatio(const double& aAspect);

            void SetSystem( KGSpace* aSpace );
            void AddSurface( KGSurface* aSurface );
            void AddSpace( KGSpace* aSpace );
            void SetSymmetry( const unsigned int& aSymmetry );

        private:
            string fDirectory;
            unsigned int fHashMaskedBits;
            double fHashThreshold;
            double fMinimumElementArea;
            double fMaximumElementAspectRatio;
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
                    virtual ~Visitor()
                    {
                    }

                    void Preprocessing( bool choice )
                    {
                        fPreprocessing = choice;
                    }
                    void Postprocessing( bool choice )
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

                    virtual void Visit( KSFieldElectrostatic& ) = 0;

                protected:
                    bool fPreprocessing;
                    bool fPostprocessing;
            };

            class VTKViewer :
                public KSFieldElectrostatic::Visitor
            {
                public:
                    VTKViewer();
                    virtual ~VTKViewer();

                    void ViewGeometry( bool choice )
                    {
                        fViewGeometry = choice;
                    }
                    void SaveGeometry( bool choice )
                    {
                        fSaveGeometry = choice;
                    }
                    void SetFile( string file )
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

                    void Visit( KSFieldElectrostatic& );

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
                    virtual bool FindSolution( double threshold, KSurfaceContainer& container );
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


            class ExplicitSuperpositionSolutionComponent
            {
                public:
                    ExplicitSuperpositionSolutionComponent(){};
                    virtual ~ExplicitSuperpositionSolutionComponent(){};

                    std::string name;
                    double scale;
                    std::string hash;
            };

            class ExplicitSuperpositionCachedBEMSolver :
                public BEMSolver
            {
                public:
                    ExplicitSuperpositionCachedBEMSolver();
                    virtual ~ExplicitSuperpositionCachedBEMSolver();

                    void SetName( std::string s )
                    {
                        fName = s;
                    }

                    void Initialize( KSurfaceContainer& container );

                    void AddSolutionComponent(ExplicitSuperpositionSolutionComponent* component)
                    {
                        fNames.push_back(component->name);
                        fScaleFactors.push_back(component->scale);
                        fHashLabels.push_back(component->hash);
                    }

                private:

                    std::string fName;
                    std::vector< std::string > fNames;
                    std::vector< double > fScaleFactors;
                    std::vector< std::string> fHashLabels;
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

            class FieldSolver
            {
                public:
                    FieldSolver();
                    virtual ~FieldSolver();

                    virtual void Initialize( KSurfaceContainer& container ) = 0;

                    virtual double Potential( const KPosition& P ) const = 0;
                    virtual KEMThreeVector ElectricField( const KPosition& P ) const = 0;
                    virtual std::pair<KEMThreeVector, double> ElectricFieldAndPotential( const KPosition& P ) const = 0;

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
                    std::pair<KEMThreeVector, double> ElectricFieldAndPotential( const KPosition& P ) const;

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
                    std::pair<KEMThreeVector, double> ElectricFieldAndPotential( const KPosition& P ) const;

                    bool UseCentralExpansion( const KPosition& P );
                    bool UseRemoteExpansion( const KPosition& P );

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
                    std::pair<KEMThreeVector, double> ElectricFieldAndPotential( const KPosition& P ) const;

                    KFMElectrostaticParameters* GetParameters()
                    {
                        return &fParameters;
                    }
                    ;

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
                fVisitors.push_back( visitor );
                return;
            }
            vector< Visitor* >& GetVisitors()
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
            vector< Visitor* > fVisitors;
            BEMSolver* fBEMSolver;
            FieldSolver* fFieldSolver;
    };
}

#endif
