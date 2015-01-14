#include "KSTrajTrajectoryExactBuilder.h"
#include "KSTrajIntegratorRK54Builder.h"
#include "KSTrajIntegratorRK65Builder.h"
#include "KSTrajIntegratorRK86Builder.h"
#include "KSTrajIntegratorRK87Builder.h"
#include "KSTrajIntegratorRK8Builder.h"
#include "KSTrajInterpolatorFastBuilder.h"
#include "KSTrajTermPropagationBuilder.h"
#include "KSTrajTermSynchrotronBuilder.h"
#include "KSTrajControlTimeBuilder.h"
#include "KSTrajControlLengthBuilder.h"
#include "KSTrajControlEnergyBuilder.h"
#include "KSTrajControlCyclotronBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTrajTrajectoryExactBuilder::~KComplexElement()
    {
    }

    static int sKSTrajTrajectoryExactStructure =
        KSTrajTrajectoryExactBuilder::Attribute< string >( "name" ) +
        KSTrajTrajectoryExactBuilder::ComplexElement< KSTrajIntegratorRK54 >( "integrator_rk54" ) +
        KSTrajTrajectoryExactBuilder::ComplexElement< KSTrajIntegratorRK65 >( "integrator_rk65" ) +
        KSTrajTrajectoryExactBuilder::ComplexElement< KSTrajIntegratorRK86 >( "integrator_rk86" ) +
        KSTrajTrajectoryExactBuilder::ComplexElement< KSTrajIntegratorRK87 >( "integrator_rk87" ) +
        KSTrajTrajectoryExactBuilder::ComplexElement< KSTrajIntegratorRK8 >( "integrator_rk8" ) +
        KSTrajTrajectoryExactBuilder::ComplexElement< KSTrajInterpolatorFast >( "interpolator_fast" ) +
        KSTrajTrajectoryExactBuilder::ComplexElement< KSTrajTermPropagation >( "term_propagation" ) +
        KSTrajTrajectoryExactBuilder::ComplexElement< KSTrajTermSynchrotron >( "term_synchrotron" ) +
        KSTrajTrajectoryExactBuilder::ComplexElement< KSTrajControlTime >( "control_time" ) +
        KSTrajTrajectoryExactBuilder::ComplexElement< KSTrajControlLength >( "control_length" ) +
        KSTrajTrajectoryExactBuilder::ComplexElement< KSTrajControlEnergy >( "control_energy" ) +
        KSTrajTrajectoryExactBuilder::ComplexElement< KSTrajControlCyclotron >( "control_cyclotron" );

    static int sKSTrajTrajectoryExact =
        KSRootBuilder::ComplexElement< KSTrajTrajectoryExact >( "kstraj_trajectory_exact" );

}
