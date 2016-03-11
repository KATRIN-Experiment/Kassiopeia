#include "KSFieldElectromagnet.h"
#include "KSFieldsMessage.h"

#include "KFile.h"

#include <limits>

#include <iostream>

namespace Kassiopeia
{

    KSFieldElectromagnet::KSFieldElectromagnet() :
            fDirectory( KEMFileInterface::GetInstance()->ActiveDirectory() ),
            fFile(),
            fSystem( NULL ),
            fConverter( NULL ),
            fContainer( NULL ),
            fFieldSolver( NULL )
    {
        fFile = KEMFileInterface::GetInstance()->GetActiveFileName();
        fFile = fFile.substr( fFile.find_last_of( "/" ) + 1, std::string::npos );
    }
    KSFieldElectromagnet::KSFieldElectromagnet( const KSFieldElectromagnet& /*aCopy*/) :
            KSComponent(),
            fDirectory( KEMFileInterface::GetInstance()->ActiveDirectory() ),
            fFile(),
            fSystem( NULL ),
            fConverter( NULL ),
            fContainer( NULL ),
            fFieldSolver( NULL )
    {
        fFile = KEMFileInterface::GetInstance()->GetActiveFileName();
        fFile = fFile.substr( fFile.find( "/" ) + 1, std::string::npos );
    }
    KSFieldElectromagnet* KSFieldElectromagnet::Clone() const
    {
        return new KSFieldElectromagnet( *this );
    }
    KSFieldElectromagnet::~KSFieldElectromagnet()
    {
    	delete fFieldSolver;
    }
    void KSFieldElectromagnet::CalculatePotential( const KThreeVector& aSamplePoint, const double& /*aSampleTime*/, KThreeVector& aPotential )
    {
        aPotential = fFieldSolver->VectorPotential( fConverter->GlobalToInternalPosition( aSamplePoint ) );
        return;
    }
    void KSFieldElectromagnet::CalculateField( const KThreeVector& aSamplePoint, const double& /*aSampleTime*/, KThreeVector& aField )
    {
        aField = fConverter->InternalToGlobalVector( fFieldSolver->MagneticField( fConverter->GlobalToInternalPosition( aSamplePoint ) ) );
        fieldmsg_debug( "magnetic field at " <<aSamplePoint<< " is " << aField <<eom);
        return;
    }
    void KSFieldElectromagnet::CalculateGradient( const KThreeVector& aSamplePoint, const double& /*aSampleTime*/, KThreeMatrix& aGradient )
    {
        aGradient = fConverter->InternalTensorToGlobal( fFieldSolver->MagneticFieldGradient( fConverter->GlobalToInternalPosition( aSamplePoint ) ) );
        fieldmsg_debug( "magnetic field gradient at " <<aSamplePoint<< " is " << aGradient <<eom);
        return;
    }

    void KSFieldElectromagnet::CalculateFieldAndGradient( const KThreeVector& aSamplePoint, const double& /*aSampleTime*/, KThreeVector& aField, KThreeMatrix& aGradient )
    {
        std::pair<KEMThreeVector, KGradient> tFieldAndGradient = fFieldSolver->MagneticFieldAndGradient( fConverter->GlobalToInternalPosition( aSamplePoint ) );
        aField = fConverter->InternalToGlobalVector( tFieldAndGradient.first );
        aGradient = fConverter->InternalTensorToGlobal( tFieldAndGradient.second );
        fieldmsg_debug( "magnetic field at " <<aSamplePoint<< " is " << aField <<eom);
        fieldmsg_debug( "magnetic field gradient at " <<aSamplePoint<< " is " << aGradient <<eom);
        return;
    }

    void KSFieldElectromagnet::SetDirectory( const string& aDirectory )
    {
        fDirectory = aDirectory;
        return;
    }
    void KSFieldElectromagnet::SetFile( const string& aFile )
    {
        fFile = aFile;
        return;
    }
    void KSFieldElectromagnet::SetSystem( KGSpace* aSpace )
    {
        fSystem = aSpace;
        return;
    }
    void KSFieldElectromagnet::AddSurface( KGSurface* aSurface )
    {
        fSurfaces.push_back( aSurface );
        return;
    }
    void KSFieldElectromagnet::AddSpace( KGSpace* aSpace )
    {
        fSpaces.push_back( aSpace );
        return;
    }

    KSFieldElectromagnet::FieldSolver::FieldSolver() :
    		fInitialized( false )
    {
    }
    KSFieldElectromagnet::FieldSolver::~FieldSolver()
    {
    }

    KSFieldElectromagnet::IntegratingFieldSolver::IntegratingFieldSolver() :
            fIntegratingFieldSolver( NULL )
    {
    }
    KSFieldElectromagnet::IntegratingFieldSolver::~IntegratingFieldSolver()
    {
    }
    void KSFieldElectromagnet::IntegratingFieldSolver::Initialize( KElectromagnetContainer& container )
    {
    	if ( fInitialized ) return;
		fIntegratingFieldSolver = new KIntegratingFieldSolver< KElectromagnetIntegrator >( container, fIntegrator );
    }
    void KSFieldElectromagnet::IntegratingFieldSolver::Deinitialize()
    {
        delete fIntegratingFieldSolver;
    }
    KEMThreeVector KSFieldElectromagnet::IntegratingFieldSolver::VectorPotential( const KPosition& P ) const
    {
        return fIntegratingFieldSolver->VectorPotential( P );
    }
    KEMThreeVector KSFieldElectromagnet::IntegratingFieldSolver::MagneticField( const KPosition& P ) const
    {
        return fIntegratingFieldSolver->MagneticField( P );
    }

    KGradient KSFieldElectromagnet::IntegratingFieldSolver::MagneticFieldGradient( const KPosition& P ) const
    {
        return fIntegratingFieldSolver->MagneticFieldGradient( P );
    }

    std::pair<KEMThreeVector, KGradient> KSFieldElectromagnet::IntegratingFieldSolver::MagneticFieldAndGradient( const KPosition& P ) const
    {
        return std::make_pair(fIntegratingFieldSolver->MagneticField( P ),fIntegratingFieldSolver->MagneticFieldGradient( P ));
    }

    KSFieldElectromagnet::ZonalHarmonicFieldSolver::ZonalHarmonicFieldSolver() :
            fZHContainer( NULL ),
            fZonalHarmonicFieldSolver( NULL )
    {
        fParameters = new KZonalHarmonicParameters();
    }
    KSFieldElectromagnet::ZonalHarmonicFieldSolver::~ZonalHarmonicFieldSolver()
    {
    	delete fParameters;
    }
    void KSFieldElectromagnet::ZonalHarmonicFieldSolver::Initialize( KElectromagnetContainer& container )
    {
    	if ( fInitialized ) return;
        // compute hash of the solved geometry
        KMD5HashGenerator solutionHashGenerator;
        string solutionHash = solutionHashGenerator.GenerateHash( container );

        fieldmsg_debug( "<shape+boundary+solution> hash is <" << solutionHash << ">" << eom )

        // compute hash of the parameter values on the bare geometry
        KMD5HashGenerator parameterHashGenerator;
        string parameterHash = parameterHashGenerator.GenerateHash( *fParameters );

        fieldmsg_debug( "<parameter> hash is <" << parameterHash << ">" << eom )

        // create label set for zh container object
        string zhContainerBase( KZonalHarmonicContainer< KMagnetostaticBasis >::Name() );
        string zhContainerName = zhContainerBase + string( "_" ) + solutionHash + string( "_" ) + parameterHash;
        vector< string > zhContainerLabels;
        zhContainerLabels.push_back( zhContainerBase );
        zhContainerLabels.push_back( solutionHash );
        zhContainerLabels.push_back( parameterHash );

        KZonalHarmonicParameters* tParametersCopy = new KZonalHarmonicParameters;
        *tParametersCopy = *fParameters;

        fZHContainer = new KZonalHarmonicContainer< KMagnetostaticBasis >( container, tParametersCopy );

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

        fZonalHarmonicFieldSolver = new KZonalHarmonicFieldSolver< KMagnetostaticBasis >( *fZHContainer, fIntegrator );
        fZonalHarmonicFieldSolver->Initialize();

        return;
    }
    void KSFieldElectromagnet::ZonalHarmonicFieldSolver::Deinitialize()
    {
    	delete fZHContainer;
    	delete fZonalHarmonicFieldSolver;
    	return;
    }

    KEMThreeVector KSFieldElectromagnet::ZonalHarmonicFieldSolver::VectorPotential( const KPosition& P ) const
    {
        return fZonalHarmonicFieldSolver->VectorPotential( P );
    }

    KEMThreeVector KSFieldElectromagnet::ZonalHarmonicFieldSolver::MagneticField( const KPosition& P ) const
    {
        return fZonalHarmonicFieldSolver->MagneticField( P );
    }

    KGradient KSFieldElectromagnet::ZonalHarmonicFieldSolver::MagneticFieldGradient( const KPosition& P ) const
    {
        return fZonalHarmonicFieldSolver->MagneticFieldGradient( P );
    }

    std::pair<KEMThreeVector, KGradient> KSFieldElectromagnet::ZonalHarmonicFieldSolver::MagneticFieldAndGradient( const KPosition& P ) const
    {
        return fZonalHarmonicFieldSolver->MagneticFieldAndGradient( P );
    }

    void KSFieldElectromagnet::InitializeComponent()
    {
        if( !fFieldSolver )
        {
            fieldmsg( eError ) << "tried to initialize electromagnet field solver <" << GetName() << "> without field solver set" << eom;
            return;
        }

        KEMFileInterface::GetInstance()->ActiveDirectory( fDirectory );
        KEMFileInterface::GetInstance()->ActiveFile( KEMFileInterface::GetInstance()->ActiveDirectory() + "/" + fFile );

        fContainer = new KElectromagnetContainer();

        fConverter = new KGElectromagnetConverter();

        fConverter->SetElectromagnetContainer( fContainer );

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
            fieldmsg( eError ) << "electromagnet field solver <" << GetName() << "> has zero surface elements" << eom;
        }

        fFieldSolver->Initialize( *fContainer );

        return;
    }

    void KSFieldElectromagnet::DeinitializeComponent()
    {
        fFieldSolver->Deinitialize();
        delete fConverter;
        delete fContainer;
    }

}
