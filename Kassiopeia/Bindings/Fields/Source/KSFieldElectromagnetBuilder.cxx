#include "KSFieldElectromagnetBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSElectromagnetZonalHarmonicSolverBuilder::~KComplexElement()
    {
    }

    template< >
    KSElectromagnetIntegratingSolverBuilder::~KComplexElement()
    {
    }

    template< >
    KSFieldElectromagnetBuilder::~KComplexElement()
    {
    }

    STATICINT sKSElectromagnetZonalHarmonicSolverStructure =
        KSElectromagnetZonalHarmonicSolverBuilder::Attribute< int >( "number_of_bifurcations" ) +
        KSElectromagnetZonalHarmonicSolverBuilder::Attribute< double >( "convergence_ratio" ) +
        KSElectromagnetZonalHarmonicSolverBuilder::Attribute< double >( "proximity_to_sourcepoint" ) +
        KSElectromagnetZonalHarmonicSolverBuilder::Attribute< double >( "convergence_parameter" ) +
        KSElectromagnetZonalHarmonicSolverBuilder::Attribute< double >( "coaxiality_tolerance" ) +
        KSElectromagnetZonalHarmonicSolverBuilder::Attribute< int >( "number_of_central_coefficients" ) +
        KSElectromagnetZonalHarmonicSolverBuilder::Attribute< bool >( "use_fractional_central_sourcepoint_spacing" ) +
        KSElectromagnetZonalHarmonicSolverBuilder::Attribute< double >( "central_sourcepoint_fractional_distance" ) +
        KSElectromagnetZonalHarmonicSolverBuilder::Attribute< double >( "central_sourcepoint_spacing" ) +
        KSElectromagnetZonalHarmonicSolverBuilder::Attribute< double >( "central_sourcepoint_start" ) +
        KSElectromagnetZonalHarmonicSolverBuilder::Attribute< double >( "central_sourcepoint_end" ) +
        KSElectromagnetZonalHarmonicSolverBuilder::Attribute< int >( "number_of_remote_coefficients" ) +
        KSElectromagnetZonalHarmonicSolverBuilder::Attribute< double >( "remote_sourcepoint_start" ) +
        KSElectromagnetZonalHarmonicSolverBuilder::Attribute< double >( "remote_sourcepoint_end" );

    STATICINT sKSFieldElectromagnetStructure =
        KSFieldElectromagnetBuilder::Attribute< string >( "name" ) +
        KSFieldElectromagnetBuilder::Attribute< string >( "file" ) +
        KSFieldElectromagnetBuilder::Attribute< string >( "directory" ) +
        KSFieldElectromagnetBuilder::Attribute< string >( "system" ) +
        KSFieldElectromagnetBuilder::Attribute< string >( "surfaces" ) +
        KSFieldElectromagnetBuilder::Attribute< string >( "spaces" ) +
        KSFieldElectromagnetBuilder::ComplexElement< KSFieldElectromagnet::IntegratingFieldSolver >( "integrating_field_solver" ) +
        KSFieldElectromagnetBuilder::ComplexElement< KSFieldElectromagnet::ZonalHarmonicFieldSolver >( "zonal_harmonic_field_solver" );

    STATICINT sKSFieldElectromagnet =
        KSRootBuilder::ComplexElement< KSFieldElectromagnet >( "ksfield_electromagnet" );

}
