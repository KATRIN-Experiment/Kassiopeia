#ifndef Kassiopeia_KSFieldElectromagnetBuilder_h_
#define Kassiopeia_KSFieldElectromagnetBuilder_h_

#include "KComplexElement.hh"
#include "KSFieldElectromagnet.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSFieldElectromagnet::IntegratingFieldSolver > KSElectromagnetIntegratingSolverBuilder;

    typedef KComplexElement< KSFieldElectromagnet::ZonalHarmonicFieldSolver > KSElectromagnetZonalHarmonicSolverBuilder;

    template< >
    inline bool KSElectromagnetZonalHarmonicSolverBuilder::AddAttribute( KContainer* aContainer )
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

    typedef KComplexElement< KSFieldElectromagnet > KSFieldElectromagnetBuilder;

    template< >
    inline bool KSFieldElectromagnetBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectromagnet::SetName );
            return true;
        }
        if( aContainer->GetName() == "directory" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectromagnet::SetDirectory );
            return true;
        }
        if( aContainer->GetName() == "file" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectromagnet::SetFile );
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
        return false;
    }

    template< >
    inline bool KSFieldElectromagnetBuilder::AddElement( KContainer* anElement )
    {
        if( anElement->GetName() == "integrating_field_solver" )
        {
            anElement->ReleaseTo( fObject, &KSFieldElectromagnet::SetFieldSolver );
            return true;
        }
        if( anElement->GetName() == "zonal_harmonic_field_solver" )
        {
            anElement->ReleaseTo( fObject, &KSFieldElectromagnet::SetFieldSolver );
            return true;
        }
        return false;
    }

}

#endif
