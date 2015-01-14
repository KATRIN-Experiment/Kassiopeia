#ifndef Kassiopeia_KSFieldElectrostaticBuilder_h_
#define Kassiopeia_KSFieldElectrostaticBuilder_h_

#include "KComplexElement.hh"
#include "KSFieldElectrostatic.h"
#include "KSFieldsMessage.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSFieldElectrostatic::VTKViewer > KSKEMFieldVTKViewerBuilder;

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

    typedef KComplexElement< KSFieldElectrostatic::CachedBEMSolver > KSCachedBEMSolverBuilder;

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

    typedef KComplexElement< KSFieldElectrostatic::RobinHoodBEMSolver > KSRobinHoodBEMSolverBuilder;

    template< >
    inline bool KSRobinHoodBEMSolverBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "tolerance" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectrostatic::RobinHoodBEMSolver::SetTolerance );
            return true;
        }
        if( aContainer->GetName() == "check_sub_interval" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectrostatic::RobinHoodBEMSolver::SetCheckSubInterval );
            return true;
        }
        if( aContainer->GetName() == "display_interval" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectrostatic::RobinHoodBEMSolver::SetDisplayInterval );
            return true;
        }
        if( aContainer->GetName() == "write_interval" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectrostatic::RobinHoodBEMSolver::SetWriteInterval );
            return true;
        }
        if( aContainer->GetName() == "plot_interval" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectrostatic::RobinHoodBEMSolver::SetPlotInterval );
            return true;
        }
        if( aContainer->GetName() == "cache_matrix_elements" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectrostatic::RobinHoodBEMSolver::CacheMatrixElements );
            return true;
        }
        if( aContainer->GetName() == "use_opencl" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectrostatic::RobinHoodBEMSolver::UseOpenCL );
            return true;
        }
        if( aContainer->GetName() == "use_vtk" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectrostatic::RobinHoodBEMSolver::UseVTK );
            return true;
        }
        return false;
    }


    typedef KComplexElement< KSFieldElectrostatic::FastMultipoleBEMSolver > KSFastMultipoleBEMSolverBuilder;

    template< >
    inline bool KSFastMultipoleBEMSolverBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "tolerance" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectrostatic::FastMultipoleBEMSolver::SetTolerance );
            return true;
        }
        if( aContainer->GetName() == "krylov_solver_type" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectrostatic::FastMultipoleBEMSolver::SetKrylovSolverType );
            return true;
        }
        if( aContainer->GetName() == "restart_cycle_size" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectrostatic::FastMultipoleBEMSolver::SetRestartCycleSize );
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
            aContainer->CopyTo( fObject, &KSFieldElectrostatic::FastMultipoleBEMSolver::UseOpenCL );
            return true;
        }
        if( aContainer->GetName() == "use_vtk" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectrostatic::FastMultipoleBEMSolver::UseVTK );
            return true;
        }
        return false;
    }

    typedef KComplexElement< KSFieldElectrostatic::GaussianEliminationBEMSolver > KSGaussianEliminationBEMSolverBuilder;

    typedef KComplexElement< KSFieldElectrostatic::IntegratingFieldSolver > KSElectrostaticIntegratingFieldSolverBuilder;

    template< >
    inline bool KSElectrostaticIntegratingFieldSolverBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "use_opencl" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectrostatic::IntegratingFieldSolver::UseOpenCL );
            return true;
        }
        return false;
    }

    typedef KComplexElement< KSFieldElectrostatic::ZonalHarmonicFieldSolver > KSElectrostaticZonalHarmonicFieldSolverBuilder;

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



    typedef KComplexElement< KSFieldElectrostatic::FastMultipoleFieldSolver > KSFastMultipoleFieldSolverBuilder;

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



    typedef KComplexElement< KSFieldElectrostatic > KSFieldElectrostaticBuilder;

    template< >
    inline bool KSFieldElectrostaticBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectrostatic::SetName );
            return true;
        }
        if( aContainer->GetName() == "directory" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectrostatic::SetDirectory );
            return true;
        }
        if( aContainer->GetName() == "file" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectrostatic::SetFile );
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
                fObject->SetSymmetry( KSFieldElectrostatic::sNoSymmetry );
                return true;
            }
            if( aContainer->AsReference< string >() == "axial" )
            {
                fObject->SetSymmetry( KSFieldElectrostatic::sAxialSymmetry );
                return true;
            }
            if( aContainer->AsReference< string >() == "discrete_axial" )
            {
                fObject->SetSymmetry( KSFieldElectrostatic::sDiscreteAxialSymmetry );
                return true;
            }
            fieldmsg( eWarning ) << "symmetry must be <none>, <axial>, or <discrete_axial>" << eom;
            return false;
        }
        if( aContainer->GetName() == "hash_masked_bits" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectrostatic::SetHashMaskedBits );
            return true;
        }
        if( aContainer->GetName() == "hash_threshold" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectrostatic::SetHashThreshold );
            return true;
        }
        return false;
    }

    typedef KComplexElement< KSFieldElectrostatic > KSFieldElectrostaticBuilder;

    template< >
    inline bool KSFieldElectrostaticBuilder::AddElement( KContainer* anElement )
    {
        if( anElement->GetName() == "viewer" )
        {
            anElement->ReleaseTo( fObject, &KSFieldElectrostatic::AddVisitor );
            return true;
        }
        if( anElement->GetName() == "cached_bem_solver" )
        {
            anElement->ReleaseTo( fObject, &KSFieldElectrostatic::SetBEMSolver );
            return true;
        }
        if( anElement->GetName() == "gaussian_elimination_bem_solver" )
        {
            anElement->ReleaseTo( fObject, &KSFieldElectrostatic::SetBEMSolver );
            return true;
        }
        if( anElement->GetName() == "robin_hood_bem_solver" )
        {
            anElement->ReleaseTo( fObject, &KSFieldElectrostatic::SetBEMSolver );
            return true;
        }
        if( anElement->GetName() == "fast_multipole_bem_solver" )
        {
            anElement->ReleaseTo( fObject, &KSFieldElectrostatic::SetBEMSolver );
            return true;
        }
        if( anElement->GetName() == "integrating_field_solver" )
        {
            anElement->ReleaseTo( fObject, &KSFieldElectrostatic::SetFieldSolver );
            return true;
        }
        if( anElement->GetName() == "zonal_harmonic_field_solver" )
        {
            anElement->ReleaseTo( fObject, &KSFieldElectrostatic::SetFieldSolver );
            return true;
        }
        if( anElement->GetName() == "fast_multipole_field_solver" )
        {
            anElement->ReleaseTo( fObject, &KSFieldElectrostatic::SetFieldSolver );
            return true;
        }
        return false;
    }

}

#endif
