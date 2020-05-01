#include "KSTrajTrajectoryExactSpinBuilder.h"

#include "KSRootBuilder.h"
#include "KSTrajControlBChangeBuilder.h"
#include "KSTrajControlCyclotronBuilder.h"
#include "KSTrajControlEnergyBuilder.h"
#include "KSTrajControlLengthBuilder.h"
#include "KSTrajControlMagneticMoment.h"
#include "KSTrajControlMomentumNumericalError.h"
#include "KSTrajControlPositionNumericalError.h"
#include "KSTrajControlSpinPrecessionBuilder.h"
#include "KSTrajControlTimeBuilder.h"
#include "KSTrajIntegratorRK54Builder.h"
#include "KSTrajIntegratorRK65Builder.h"
#include "KSTrajIntegratorRK86Builder.h"
#include "KSTrajIntegratorRK87Builder.h"
#include "KSTrajIntegratorRK8Builder.h"
#include "KSTrajIntegratorRKDP54Builder.h"
#include "KSTrajIntegratorRKDP853Builder.h"
#include "KSTrajInterpolatorContinuousRungeKuttaBuilder.h"
#include "KSTrajInterpolatorFastBuilder.h"
#include "KSTrajInterpolatorHermiteBuilder.h"
#include "KSTrajTermGravityBuilder.h"
#include "KSTrajTermPropagationBuilder.h"
#include "KSTrajTermSynchrotronBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTrajTrajectoryExactSpinBuilder::~KComplexElement() {}

STATICINT sKSTrajTrajectoryExactSpinStructure =
    KSTrajTrajectoryExactSpinBuilder::Attribute<string>("name") +
    KSTrajTrajectoryExactSpinBuilder::Attribute<unsigned int>("attempt_limit") +
    KSTrajTrajectoryExactSpinBuilder::ComplexElement<KSTrajIntegratorRK54>("integrator_rk54") +
    KSTrajTrajectoryExactSpinBuilder::ComplexElement<KSTrajIntegratorRKDP54>("integrator_rkdp54") +
    KSTrajTrajectoryExactSpinBuilder::ComplexElement<KSTrajIntegratorRKDP853>("integrator_rkdp853") +
    KSTrajTrajectoryExactSpinBuilder::ComplexElement<KSTrajIntegratorRK65>("integrator_rk65") +
    KSTrajTrajectoryExactSpinBuilder::ComplexElement<KSTrajIntegratorRK86>("integrator_rk86") +
    KSTrajTrajectoryExactSpinBuilder::ComplexElement<KSTrajIntegratorRK87>("integrator_rk87") +
    KSTrajTrajectoryExactSpinBuilder::ComplexElement<KSTrajIntegratorRK8>("integrator_rk8") +
    KSTrajTrajectoryExactSpinBuilder::ComplexElement<KSTrajInterpolatorFast>("interpolator_fast") +
    KSTrajTrajectoryExactSpinBuilder::ComplexElement<KSTrajInterpolatorHermite>("interpolator_hermite") +
    KSTrajTrajectoryExactSpinBuilder::ComplexElement<KSTrajInterpolatorContinuousRungeKutta>("interpolator_crk") +
    KSTrajTrajectoryExactSpinBuilder::ComplexElement<KSTrajTermPropagation>("term_propagation") +
    KSTrajTrajectoryExactSpinBuilder::ComplexElement<KSTrajTermSynchrotron>("term_synchrotron") +
    KSTrajTrajectoryExactSpinBuilder::ComplexElement<KSTrajTermGravity>("term_gravity") +
    KSTrajTrajectoryExactSpinBuilder::ComplexElement<KSTrajControlTime>("control_time") +
    KSTrajTrajectoryExactSpinBuilder::ComplexElement<KSTrajControlLength>("control_length") +
    KSTrajTrajectoryExactSpinBuilder::ComplexElement<KSTrajControlEnergy>("control_energy") +
    KSTrajTrajectoryExactSpinBuilder::ComplexElement<KSTrajControlPositionNumericalError>("control_position_error") +
    KSTrajTrajectoryExactSpinBuilder::ComplexElement<KSTrajControlMomentumNumericalError>("control_momentum_error") +
    KSTrajTrajectoryExactSpinBuilder::ComplexElement<KSTrajControlMagneticMoment>("control_magnetic_moment") +
    KSTrajTrajectoryExactSpinBuilder::ComplexElement<KSTrajControlCyclotron>("control_cyclotron") +
    KSTrajTrajectoryExactSpinBuilder::ComplexElement<KSTrajControlSpinPrecession>("control_spin_precession") +
    KSTrajTrajectoryExactSpinBuilder::ComplexElement<KSTrajControlBChange>("control_B_change") +
    KSTrajTrajectoryExactSpinBuilder::Attribute<double>("piecewise_tolerance") +
    KSTrajTrajectoryExactSpinBuilder::Attribute<unsigned int>("max_segments");

STATICINT sKSTrajTrajectoryExactSpin =
    KSRootBuilder::ComplexElement<KSTrajTrajectoryExactSpin>("kstraj_trajectory_exact_spin");

}  // namespace katrin
