#include "KSTrajTrajectoryElectricBuilder.h"
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
#include "KSTrajControlTimeBuilder.h"
#include "KSTrajControlLengthBuilder.h"
#include "KSTrajControlBChangeBuilder.h"
#include "KSRootBuilder.h"


using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSTrajTrajectoryElectricBuilder::~KComplexElement()
    {
    }

    STATICINT sKSTrajTrajectoryElectricStructure =
        KSTrajTrajectoryElectricBuilder::Attribute< string >( "name" ) +
        KSTrajTrajectoryElectricBuilder::Attribute< unsigned int >( "attempt_limit" ) +
        KSTrajTrajectoryElectricBuilder::ComplexElement< KSTrajIntegratorRK54 >( "integrator_rk54" ) +
        KSTrajTrajectoryElectricBuilder::ComplexElement< KSTrajIntegratorRKDP54 >( "integrator_rkdp54" ) +
        KSTrajTrajectoryElectricBuilder::ComplexElement< KSTrajIntegratorRKDP853 >( "integrator_rkdp853" ) +
        KSTrajTrajectoryElectricBuilder::ComplexElement< KSTrajIntegratorRK65 >( "integrator_rk65" ) +
        KSTrajTrajectoryElectricBuilder::ComplexElement< KSTrajIntegratorRK86 >( "integrator_rk86" ) +
        KSTrajTrajectoryElectricBuilder::ComplexElement< KSTrajIntegratorRK87 >( "integrator_rk87" ) +
        KSTrajTrajectoryElectricBuilder::ComplexElement< KSTrajIntegratorRK8 >( "integrator_rk8" ) +
        KSTrajTrajectoryElectricBuilder::ComplexElement< KSTrajInterpolatorFast >( "interpolator_fast" ) +
        KSTrajTrajectoryElectricBuilder::ComplexElement< KSTrajInterpolatorHermite >( "interpolator_hermite" ) +
        KSTrajTrajectoryElectricBuilder::ComplexElement< KSTrajInterpolatorContinuousRungeKutta >( "interpolator_crk" ) +
        KSTrajTrajectoryElectricBuilder::ComplexElement< KSTrajTermPropagation >( "term_propagation" ) +
        KSTrajTrajectoryElectricBuilder::ComplexElement< KSTrajControlTime >( "control_time" ) +
        KSTrajTrajectoryElectricBuilder::ComplexElement< KSTrajControlLength >( "control_length" ) +
        KSTrajTrajectoryElectricBuilder::ComplexElement< KSTrajControlBChange >( "control_B_change" ) +
        KSTrajTrajectoryElectricBuilder::Attribute< double >( "piecewise_tolerance" ) +
        KSTrajTrajectoryElectricBuilder::Attribute< unsigned int >( "max_segments" );

    STATICINT sToolboxKSTrajTrajectoryElectric =
        KSRootBuilder::ComplexElement< KSTrajTrajectoryElectric >( "kstraj_trajectory_electric" );

}
