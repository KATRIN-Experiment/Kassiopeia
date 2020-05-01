#include "KSTrajTrajectoryExactTrappedBuilder.h"

#include "KSTrajIntegratorRK8Builder.h"
#include "KSTrajIntegratorSym4Builder.h"
//#include "KSTrajInterpolatorFastBuilder.h"
//#include "KSTrajInterpolatorHermiteBuilder.h"
//#include "KSTrajInterpolatorContinuousRungeKuttaBuilder.h"
#include "KSRootBuilder.h"
#include "KSTrajControlLengthBuilder.h"
#include "KSTrajControlTimeBuilder.h"
#include "KSTrajTermPropagationBuilder.h"
#include "KSTrajTermSynchrotronBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTrajTrajectoryExactTrappedBuilder::~KComplexElement() {}

STATICINT sKSTrajTrajectoryExactTrappedStructure =
    KSTrajTrajectoryExactTrappedBuilder::Attribute<string>("name") +
    KSTrajTrajectoryExactTrappedBuilder::Attribute<unsigned int>("attempt_limit") +
    KSTrajTrajectoryExactTrappedBuilder::ComplexElement<KSTrajIntegratorSym4>("integrator_sym4") +
    KSTrajTrajectoryExactTrappedBuilder::ComplexElement<KSTrajIntegratorRK8>("integrator_rk8") +
    //KSTrajTrajectoryExactTrappedBuilder::ComplexElement< KSTrajInterpolatorFast >( "interpolator_fast" ) +
    //KSTrajTrajectoryExactTrappedBuilder::ComplexElement< KSTrajInterpolatorHermite >( "interpolator_hermite" ) +
    //KSTrajTrajectoryExactTrappedBuilder::ComplexElement< KSTrajInterpolatorContinuousRungeKutta >( "interpolator_crk" ) +
    KSTrajTrajectoryExactTrappedBuilder::ComplexElement<KSTrajTermPropagation>("term_propagation") +
    KSTrajTrajectoryExactTrappedBuilder::ComplexElement<KSTrajTermSynchrotron>("term_synchrotron") +
    KSTrajTrajectoryExactTrappedBuilder::ComplexElement<KSTrajControlTime>("control_time") +
    KSTrajTrajectoryExactTrappedBuilder::ComplexElement<KSTrajControlLength>("control_length") +
    KSTrajTrajectoryExactTrappedBuilder::Attribute<double>("piecewise_tolerance") +
    KSTrajTrajectoryExactTrappedBuilder::Attribute<unsigned int>("max_segments");

STATICINT sKSTrajTrajectoryExactTrapped =
    KSRootBuilder::ComplexElement<KSTrajTrajectoryExactTrapped>("kstraj_trajectory_exact_trapped");

}  // namespace katrin
