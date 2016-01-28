#include "KSTrajTrajectoryExactBuilder.h"
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
    KSTrajTrajectoryExactBuilder::~KComplexElement()
    {
    }

    STATICINT sKSTrajTrajectoryExactStructure =
        KSTrajTrajectoryExactBuilder::Attribute< string >( "name" ) +
        KSTrajTrajectoryExactBuilder::Attribute< unsigned int >( "attempt_limit" ) +
        KSTrajTrajectoryExactBuilder::ComplexElement< KSTrajIntegratorRK54 >( "integrator_rk54" ) +
        KSTrajTrajectoryExactBuilder::ComplexElement< KSTrajIntegratorRKDP54 >( "integrator_rkdp54" ) +
        KSTrajTrajectoryExactBuilder::ComplexElement< KSTrajIntegratorRKDP853 >( "integrator_rkdp853" ) +
        KSTrajTrajectoryExactBuilder::ComplexElement< KSTrajIntegratorRK65 >( "integrator_rk65" ) +
        KSTrajTrajectoryExactBuilder::ComplexElement< KSTrajIntegratorRK86 >( "integrator_rk86" ) +
        KSTrajTrajectoryExactBuilder::ComplexElement< KSTrajIntegratorRK87 >( "integrator_rk87" ) +
        KSTrajTrajectoryExactBuilder::ComplexElement< KSTrajIntegratorRK8 >( "integrator_rk8" ) +
        KSTrajTrajectoryExactBuilder::ComplexElement< KSTrajInterpolatorFast >( "interpolator_fast" ) +
        KSTrajTrajectoryExactBuilder::ComplexElement< KSTrajInterpolatorHermite >( "interpolator_hermite" ) +
        KSTrajTrajectoryExactBuilder::ComplexElement< KSTrajInterpolatorContinuousRungeKutta >( "interpolator_crk" ) +
        KSTrajTrajectoryExactBuilder::ComplexElement< KSTrajTermPropagation >( "term_propagation" ) +
        KSTrajTrajectoryExactBuilder::ComplexElement< KSTrajTermSynchrotron >( "term_synchrotron" ) +
        KSTrajTrajectoryExactBuilder::ComplexElement< KSTrajControlTime >( "control_time" ) +
        KSTrajTrajectoryExactBuilder::ComplexElement< KSTrajControlLength >( "control_length" ) +
        KSTrajTrajectoryExactBuilder::ComplexElement< KSTrajControlEnergy >( "control_energy" ) +
        KSTrajTrajectoryExactBuilder::ComplexElement< KSTrajControlPositionNumericalError >( "control_position_error" ) +
        KSTrajTrajectoryExactBuilder::ComplexElement< KSTrajControlMomentumNumericalError >( "control_momentum_error" ) +
        KSTrajTrajectoryExactBuilder::ComplexElement< KSTrajControlMagneticMoment >( "control_magnetic_moment" ) +
        KSTrajTrajectoryExactBuilder::ComplexElement< KSTrajControlCyclotron >( "control_cyclotron" ) +
        KSTrajTrajectoryExactBuilder::Attribute< double >( "piecewise_tolerance" ) +
        KSTrajTrajectoryExactBuilder::Attribute< unsigned int >( "max_segments" );

    STATICINT sKSTrajTrajectoryExact =
        KSRootBuilder::ComplexElement< KSTrajTrajectoryExact >( "kstraj_trajectory_exact" );

}
