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
            fObject->GetParameters()->SetNBifurcations( aContainer->AsReference<int>() );
            return true;
        }
        if( aContainer->GetName() == "convergence_ratio" )
        {
            fObject->GetParameters()->SetConvergenceRatio( aContainer->AsReference<double>() );
            return true;
        }
        if( aContainer->GetName() == "proximity_to_sourcepoint" )
        {
            fObject->GetParameters()->SetProximityToSourcePoint( aContainer->AsReference<double>() );
            return true;
        }
        if( aContainer->GetName() == "convergence_parameter" )
        {
            fObject->GetParameters()->SetConvergenceParameter( aContainer->AsReference<double>() );
            return true;
        }
        if( aContainer->GetName() == "coaxiality_tolerance" )
        {
            fObject->GetParameters()->SetCoaxialityTolerance( aContainer->AsReference<double>() );
            return true;
        }
        if( aContainer->GetName() == "number_of_central_coefficients" )
        {
            fObject->GetParameters()->SetNCentralCoefficients( aContainer->AsReference<int>() );
            return true;
        }
        if( aContainer->GetName() == "use_fractional_central_sourcepoint_spacing" )
        {
            fObject->GetParameters()->SetCentralFractionalSpacing( aContainer->AsReference<bool>() );
            return true;
        }
        if( aContainer->GetName() == "central_sourcepoint_fractional_distance" )
        {
            fObject->GetParameters()->SetCentralFractionalDistance( aContainer->AsReference<double>() );
            return true;
        }
        if( aContainer->GetName() == "central_sourcepoint_spacing" )
        {
            fObject->GetParameters()->SetCentralDeltaZ( aContainer->AsReference<double>() );
            return true;
        }
        if( aContainer->GetName() == "central_sourcepoint_start" )
        {
            fObject->GetParameters()->SetCentralZ1( aContainer->AsReference<double>() );
            return true;
        }
        if( aContainer->GetName() == "central_sourcepoint_end" )
        {
            fObject->GetParameters()->SetCentralZ2( aContainer->AsReference<double>() );
            return true;
        }
        if( aContainer->GetName() == "number_of_remote_coefficients" )
        {
            fObject->GetParameters()->SetNRemoteCoefficients( aContainer->AsReference<int>() );
            return true;
        }
        if( aContainer->GetName() == "remote_sourcepoint_start" )
        {
            fObject->GetParameters()->SetRemoteZ1( aContainer->AsReference<double>() );
            return true;
        }
        if( aContainer->GetName() == "remote_sourcepoint_end" )
        {
            fObject->GetParameters()->SetRemoteZ2( aContainer->AsReference<double>() );
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
