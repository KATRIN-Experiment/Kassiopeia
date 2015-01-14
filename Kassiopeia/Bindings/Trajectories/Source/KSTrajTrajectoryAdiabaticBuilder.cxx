#include "KSTrajTrajectoryAdiabaticBuilder.h"
#include "KSTrajIntegratorRK54Builder.h"
#include "KSTrajIntegratorRK65Builder.h"
#include "KSTrajIntegratorRK86Builder.h"
#include "KSTrajIntegratorRK87Builder.h"
#include "KSTrajIntegratorRK8Builder.h"
#include "KSTrajInterpolatorFastBuilder.h"
#include "KSTrajTermPropagationBuilder.h"
#include "KSTrajTermSynchrotronBuilder.h"
#include "KSTrajTermDriftBuilder.h"
#include "KSTrajTermGyrationBuilder.h"
#include "KSTrajControlTimeBuilder.h"
#include "KSTrajControlLengthBuilder.h"
#include "KSTrajControlEnergyBuilder.h"
#include "KSTrajControlCyclotronBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTrajTrajectoryAdiabaticBuilder::~KComplexElement()
    {
    }

    static int sKSTrajTrajectoryAdiabaticStructure =
        KSTrajTrajectoryAdiabaticBuilder::Attribute< string >( "name" ) +
        KSTrajTrajectoryAdiabaticBuilder::ComplexElement< KSTrajIntegratorRK54 >( "integrator_rk54" ) +
        KSTrajTrajectoryAdiabaticBuilder::ComplexElement< KSTrajIntegratorRK65 >( "integrator_rk65" ) +
        KSTrajTrajectoryAdiabaticBuilder::ComplexElement< KSTrajIntegratorRK86 >( "integrator_rk86" ) +
        KSTrajTrajectoryAdiabaticBuilder::ComplexElement< KSTrajIntegratorRK87 >( "integrator_rk87" ) +
        KSTrajTrajectoryAdiabaticBuilder::ComplexElement< KSTrajIntegratorRK8 >( "integrator_rk8" ) +
        KSTrajTrajectoryAdiabaticBuilder::ComplexElement< KSTrajInterpolatorFast >( "interpolator_fast" ) +
        KSTrajTrajectoryAdiabaticBuilder::ComplexElement< KSTrajTermPropagation >( "term_propagation" ) +
        KSTrajTrajectoryAdiabaticBuilder::ComplexElement< KSTrajTermSynchrotron >( "term_synchrotron" ) +
        KSTrajTrajectoryAdiabaticBuilder::ComplexElement< KSTrajTermDrift >( "term_drift" ) +
        KSTrajTrajectoryAdiabaticBuilder::ComplexElement< KSTrajTermGyration >( "term_gyration" ) +
        KSTrajTrajectoryAdiabaticBuilder::ComplexElement< KSTrajControlTime >( "control_time" ) +
        KSTrajTrajectoryAdiabaticBuilder::ComplexElement< KSTrajControlLength >( "control_length" ) +
        KSTrajTrajectoryAdiabaticBuilder::ComplexElement< KSTrajControlEnergy >( "control_energy" ) +
        KSTrajTrajectoryAdiabaticBuilder::ComplexElement< KSTrajControlCyclotron >( "control_cyclotron" );

    static int sToolboxKSTrajTrajectoryAdiabatic =
        KSRootBuilder::ComplexElement< KSTrajTrajectoryAdiabatic >( "kstraj_trajectory_adiabatic" );

}
