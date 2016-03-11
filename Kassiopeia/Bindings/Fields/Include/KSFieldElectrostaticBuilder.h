#ifndef Kassiopeia_KSFieldElectrostaticBuilder_h_
#define Kassiopeia_KSFieldElectrostaticBuilder_h_

#include "KComplexElement.hh"
#include "KSFieldElectrostatic.h"
#include "KSFieldsMessage.h"
#include "KSFieldKrylovBEMSolverBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSFieldElectrostatic::VTKViewer > KSKEMFieldVTKViewerBuilder;

    template< >
    inline bool KSKEMFieldVTKViewerBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "file" )
        {            
            aContainer->CopyTo(fObject, &KSFieldElectrostatic::VTKViewer::SetFile );
            return true;
        }
        if( aContainer->GetName() == "view" )
        {
            aContainer->CopyTo(fObject, &KSFieldElectrostatic::VTKViewer::ViewGeometry );
            return true;
        }
        if( aContainer->GetName() == "save" )
        {
            aContainer->CopyTo(fObject, &KSFieldElectrostatic::VTKViewer::SaveGeometry );

            return true;
        }
        if( aContainer->GetName() == "preprocessing" )
        {
            aContainer->CopyTo(fObject, &KSFieldElectrostatic::VTKViewer::Preprocessing );
            return true;
        }
        if( aContainer->GetName() == "postprocessing" )
        {
            aContainer->CopyTo(fObject, &KSFieldElectrostatic::VTKViewer::Postprocessing );
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
            aContainer->CopyTo(fObject, &KSFieldElectrostatic::CachedBEMSolver::SetName );
            return true;
        }
        if( aContainer->GetName() == "hash" )
        {
            aContainer->CopyTo(fObject, &KSFieldElectrostatic::CachedBEMSolver::SetHash );
            return true;
        }
        return false;
    }

    typedef KComplexElement< KSFieldElectrostatic::ExplicitSuperpositionSolutionComponent > KSExplicitSuperpositionSolutionComponentBuilder;

    template< >
    inline bool KSExplicitSuperpositionSolutionComponentBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {            
            aContainer->CopyTo( fObject->name );
            return true;
        }
        if( aContainer->GetName() == "scale" )
        {            
            aContainer->CopyTo( fObject->scale );
            return true;
        }
        if( aContainer->GetName() == "hash" )
        {            
            aContainer->CopyTo( fObject->hash );
            return true;
        }
        return false;
    }


    typedef KComplexElement< KSFieldElectrostatic::ExplicitSuperpositionCachedBEMSolver > KSExplicitSuperpositionCachedBEMSolverBuilder;

    template< >
    inline bool KSExplicitSuperpositionCachedBEMSolverBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {            
            aContainer->CopyTo( fObject, &KSFieldElectrostatic::ExplicitSuperpositionCachedBEMSolver::SetName );
            return true;
        }
        return false;
    }

    template< >
    inline bool KSExplicitSuperpositionCachedBEMSolverBuilder::AddElement( KContainer* anElement )
    {
        if( anElement->GetName() == "component" )
        {
            anElement->ReleaseTo( fObject, &KSFieldElectrostatic::ExplicitSuperpositionCachedBEMSolver::AddSolutionComponent );
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



    typedef KComplexElement< KSFieldElectrostatic::FastMultipoleFieldSolver > KSFastMultipoleFieldSolverBuilder;

    template< >
    inline bool KSFastMultipoleFieldSolverBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "top_level_divisions" )
        {
            aContainer->CopyTo( fObject->GetParameters()->top_level_divisions );
            return true;
        }
        if( aContainer->GetName() == "tree_level_divisions" )
        {
            aContainer->CopyTo( fObject->GetParameters()->divisions );
            return true;
        }
        if( aContainer->GetName() == "expansion_degree" )
        {
            aContainer->CopyTo( fObject->GetParameters()->degree );
            return true;
        }
        if( aContainer->GetName() == "neighbor_order" )
        {
            aContainer->CopyTo( fObject->GetParameters()->zeromask );
            return true;
        }
        if( aContainer->GetName() == "maximum_tree_depth" )
        {
            aContainer->CopyTo( fObject->GetParameters()->maximum_tree_depth );
            return true;
        }
        if( aContainer->GetName() == "region_expansion_factor" )
        {
            aContainer->CopyTo( fObject->GetParameters()->region_expansion_factor );
            return true;
        }
        if( aContainer->GetName() == "use_region_size_estimation" )
        {
            aContainer->CopyTo( fObject->GetParameters()->use_region_estimation );
            return true;
        }
        if( aContainer->GetName() == "world_cube_center_x" )
        {
            aContainer->CopyTo( fObject->GetParameters()->world_center_x );
            return true;
        }
        if( aContainer->GetName() == "world_cube_center_y" )
        {
            aContainer->CopyTo( fObject->GetParameters()->world_center_y );
            return true;
        }
        if( aContainer->GetName() == "world_cube_center_z" )
        {
            aContainer->CopyTo( fObject->GetParameters()->world_center_z );
            return true;
        }
        if( aContainer->GetName() == "world_cube_length" )
        {
            aContainer->CopyTo( fObject->GetParameters()->world_length );
            return true;

        }
        if( aContainer->GetName() == "use_caching" )
        {
            aContainer->CopyTo( fObject->GetParameters()->use_caching );
            return true;
        }
        if( aContainer->GetName() == "verbosity" )
        {
            aContainer->CopyTo( fObject->GetParameters()->verbosity );
            return true;
        }
        if( aContainer->GetName() == "insertion_ratio" )
        {
            aContainer->CopyTo( fObject->GetParameters()->insertion_ratio );
            return true;
        }
        if( aContainer->GetName() == "use_opencl" )
        {            
            aContainer->CopyTo(fObject, &KSFieldElectrostatic::FastMultipoleFieldSolver::UseOpenCL);
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
        if( aContainer->GetName() == "minimum_element_area" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectrostatic::SetMinimumElementArea);
            return true;
        }
        if( aContainer->GetName() == "maximum_element_aspect_ratio" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectrostatic::SetMaximumElementAspectRatio);
            return true;
        }
        return false;
    }

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
        if( anElement->GetName() == "explicit_superposition_cached_bem_solver" )
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
        if( anElement->GetName() == "krylov_bem_solver" )
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
