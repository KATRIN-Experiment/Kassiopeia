#include "KSTrajTrajectoryAdiabaticBuilder.h"
#include "KSTrajIntegratorRK54Builder.h"
#include "KSTrajIntegratorRKDP54Builder.h"
#include "KSTrajIntegratorRKDP853Builder.h"
#include "KSTrajIntegratorRK65Builder.h"
#include "KSTrajIntegratorRK86Builder.h"
#include "KSTrajIntegratorRK87Builder.h"
#include "KSTrajIntegratorRK8Builder.h"
#include "KSTrajInterpolatorFastBuilder.h"
#include "KSTrajInterpolatorHermiteBuilder.h"
#include "KSTrajInterpolatorContinuousRungeKuttaBuilder.h"
#include "KSTrajTermPropagationBuilder.h"
#include "KSTrajTermSynchrotronBuilder.h"
#include "KSTrajTermDriftBuilder.h"
#include "KSTrajTermGyrationBuilder.h"
#include "KSTrajControlTimeBuilder.h"
#include "KSTrajControlLengthBuilder.h"
#include "KSTrajControlEnergyBuilder.h"
#include "KSTrajControlMagneticMoment.h"
#include "KSTrajControlCyclotronBuilder.h"
#include "KSTrajControlPositionNumericalError.h"
#include "KSTrajControlMomentumNumericalError.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTrajTrajectoryAdiabaticBuilder::~KComplexElement()
    {
    }

    STATICINT sKSTrajTrajectoryAdiabaticStructure =
        KSTrajTrajectoryAdiabaticBuilder::Attribute< string >( "name" ) +
        KSTrajTrajectoryAdiabaticBuilder::Attribute< unsigned int >( "attempt_limit" ) +
        KSTrajTrajectoryAdiabaticBuilder::ComplexElement< KSTrajIntegratorRK54 >( "integrator_rk54" ) +
        KSTrajTrajectoryAdiabaticBuilder::ComplexElement< KSTrajIntegratorRKDP54 >( "integrator_rkdp54" ) +
        KSTrajTrajectoryAdiabaticBuilder::ComplexElement< KSTrajIntegratorRKDP853 >( "integrator_rkdp853" ) +
        KSTrajTrajectoryAdiabaticBuilder::ComplexElement< KSTrajIntegratorRK65 >( "integrator_rk65" ) +
        KSTrajTrajectoryAdiabaticBuilder::ComplexElement< KSTrajIntegratorRK86 >( "integrator_rk86" ) +
        KSTrajTrajectoryAdiabaticBuilder::ComplexElement< KSTrajIntegratorRK87 >( "integrator_rk87" ) +
        KSTrajTrajectoryAdiabaticBuilder::ComplexElement< KSTrajIntegratorRK8 >( "integrator_rk8" ) +
        KSTrajTrajectoryAdiabaticBuilder::ComplexElement< KSTrajInterpolatorFast >( "interpolator_fast" ) +
        KSTrajTrajectoryAdiabaticBuilder::ComplexElement< KSTrajInterpolatorHermite >( "interpolator_hermite" ) +
        KSTrajTrajectoryAdiabaticBuilder::ComplexElement< KSTrajInterpolatorContinuousRungeKutta >( "interpolator_crk" ) +
        KSTrajTrajectoryAdiabaticBuilder::ComplexElement< KSTrajTermPropagation >( "term_propagation" ) +
        KSTrajTrajectoryAdiabaticBuilder::ComplexElement< KSTrajTermSynchrotron >( "term_synchrotron" ) +
        KSTrajTrajectoryAdiabaticBuilder::ComplexElement< KSTrajTermDrift >( "term_drift" ) +
        KSTrajTrajectoryAdiabaticBuilder::ComplexElement< KSTrajTermGyration >( "term_gyration" ) +
        KSTrajTrajectoryAdiabaticBuilder::ComplexElement< KSTrajControlTime >( "control_time" ) +
        KSTrajTrajectoryAdiabaticBuilder::ComplexElement< KSTrajControlLength >( "control_length" ) +
        KSTrajTrajectoryAdiabaticBuilder::ComplexElement< KSTrajControlEnergy >( "control_energy" ) +
        KSTrajTrajectoryAdiabaticBuilder::ComplexElement< KSTrajControlPositionNumericalError >( "control_position_error" ) +
        KSTrajTrajectoryAdiabaticBuilder::ComplexElement< KSTrajControlMomentumNumericalError >( "control_momentum_error" ) +
        KSTrajTrajectoryAdiabaticBuilder::ComplexElement< KSTrajControlMagneticMoment >( "control_magnetic_moment" ) +
        KSTrajTrajectoryAdiabaticBuilder::ComplexElement< KSTrajControlCyclotron >( "control_cyclotron" ) +
        KSTrajTrajectoryAdiabaticBuilder::Attribute< double >( "piecewise_tolerance" ) +
        KSTrajTrajectoryAdiabaticBuilder::Attribute< unsigned int >( "max_segments" ) +
        KSTrajTrajectoryAdiabaticBuilder::Attribute< bool >( "use_true_position" ) +
        KSTrajTrajectoryAdiabaticBuilder::Attribute< double >( "cyclotron_fraction" );

    STATICINT sToolboxKSTrajTrajectoryAdiabatic =
        KSRootBuilder::ComplexElement< KSTrajTrajectoryAdiabatic >( "kstraj_trajectory_adiabatic" );

}
