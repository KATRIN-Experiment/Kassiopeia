#include "KSTrajTrajectoryAdiabaticSpinBuilder.h"

#include "KSRootBuilder.h"
#include "KSTrajControlBChangeBuilder.h"
#include "KSTrajControlCyclotronBuilder.h"
#include "KSTrajControlEnergyBuilder.h"
#include "KSTrajControlLengthBuilder.h"
#include "KSTrajControlMDotBuilder.h"
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

template<> KSTrajTrajectoryAdiabaticSpinBuilder::~KComplexElement() {}

STATICINT sKSTrajTrajectoryAdiabaticSpinStructure =
    KSTrajTrajectoryAdiabaticSpinBuilder::Attribute<string>("name") +
    KSTrajTrajectoryAdiabaticSpinBuilder::Attribute<unsigned int>("attempt_limit") +
    KSTrajTrajectoryAdiabaticSpinBuilder::ComplexElement<KSTrajIntegratorRK54>("integrator_rk54") +
    KSTrajTrajectoryAdiabaticSpinBuilder::ComplexElement<KSTrajIntegratorRKDP54>("integrator_rkdp54") +
    KSTrajTrajectoryAdiabaticSpinBuilder::ComplexElement<KSTrajIntegratorRKDP853>("integrator_rkdp853") +
    KSTrajTrajectoryAdiabaticSpinBuilder::ComplexElement<KSTrajIntegratorRK65>("integrator_rk65") +
    KSTrajTrajectoryAdiabaticSpinBuilder::ComplexElement<KSTrajIntegratorRK86>("integrator_rk86") +
    KSTrajTrajectoryAdiabaticSpinBuilder::ComplexElement<KSTrajIntegratorRK87>("integrator_rk87") +
    KSTrajTrajectoryAdiabaticSpinBuilder::ComplexElement<KSTrajIntegratorRK8>("integrator_rk8") +
    KSTrajTrajectoryAdiabaticSpinBuilder::ComplexElement<KSTrajInterpolatorFast>("interpolator_fast") +
    KSTrajTrajectoryAdiabaticSpinBuilder::ComplexElement<KSTrajInterpolatorHermite>("interpolator_hermite") +
    KSTrajTrajectoryAdiabaticSpinBuilder::ComplexElement<KSTrajInterpolatorContinuousRungeKutta>("interpolator_crk") +
    KSTrajTrajectoryAdiabaticSpinBuilder::ComplexElement<KSTrajTermPropagation>("term_propagation") +
    KSTrajTrajectoryAdiabaticSpinBuilder::ComplexElement<KSTrajTermSynchrotron>("term_synchrotron") +
    KSTrajTrajectoryAdiabaticSpinBuilder::ComplexElement<KSTrajTermGravity>("term_gravity") +
    KSTrajTrajectoryAdiabaticSpinBuilder::ComplexElement<KSTrajControlTime>("control_time") +
    KSTrajTrajectoryAdiabaticSpinBuilder::ComplexElement<KSTrajControlLength>("control_length") +
    KSTrajTrajectoryAdiabaticSpinBuilder::ComplexElement<KSTrajControlEnergy>("control_energy") +
    KSTrajTrajectoryAdiabaticSpinBuilder::ComplexElement<KSTrajControlPositionNumericalError>(
        "control_position_error") +
    KSTrajTrajectoryAdiabaticSpinBuilder::ComplexElement<KSTrajControlMomentumNumericalError>(
        "control_momentum_error") +
    KSTrajTrajectoryAdiabaticSpinBuilder::ComplexElement<KSTrajControlMagneticMoment>("control_magnetic_moment") +
    KSTrajTrajectoryAdiabaticSpinBuilder::ComplexElement<KSTrajControlCyclotron>("control_cyclotron") +
    KSTrajTrajectoryAdiabaticSpinBuilder::ComplexElement<KSTrajControlSpinPrecession>("control_spin_precession") +
    KSTrajTrajectoryAdiabaticSpinBuilder::ComplexElement<KSTrajControlMDot>("control_m_dot") +
    KSTrajTrajectoryAdiabaticSpinBuilder::ComplexElement<KSTrajControlBChange>("control_B_change") +
    KSTrajTrajectoryAdiabaticSpinBuilder::Attribute<double>("piecewise_tolerance") +
    KSTrajTrajectoryAdiabaticSpinBuilder::Attribute<unsigned int>("max_segments");

STATICINT sKSTrajTrajectoryAdiabaticSpin =
    KSRootBuilder::ComplexElement<KSTrajTrajectoryAdiabaticSpin>("kstraj_trajectory_adiabatic_spin");

}  // namespace katrin
