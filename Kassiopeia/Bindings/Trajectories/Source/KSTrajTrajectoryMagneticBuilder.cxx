#include "KSTrajTrajectoryMagneticBuilder.h"

#include "KSRootBuilder.h"
#include "KSTrajControlBChangeBuilder.h"
#include "KSTrajControlLengthBuilder.h"
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
#include "KSTrajTermPropagationBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTrajTrajectoryMagneticBuilder::~KComplexElement() {}

STATICINT sKSTrajTrajectoryMagneticStructure =
    KSTrajTrajectoryMagneticBuilder::Attribute<string>("name") +
    KSTrajTrajectoryMagneticBuilder::Attribute<unsigned int>("attempt_limit") +
    KSTrajTrajectoryMagneticBuilder::ComplexElement<KSTrajIntegratorRK54>("integrator_rk54") +
    KSTrajTrajectoryMagneticBuilder::ComplexElement<KSTrajIntegratorRKDP54>("integrator_rkdp54") +
    KSTrajTrajectoryMagneticBuilder::ComplexElement<KSTrajIntegratorRKDP853>("integrator_rkdp853") +
    KSTrajTrajectoryMagneticBuilder::ComplexElement<KSTrajIntegratorRK65>("integrator_rk65") +
    KSTrajTrajectoryMagneticBuilder::ComplexElement<KSTrajIntegratorRK86>("integrator_rk86") +
    KSTrajTrajectoryMagneticBuilder::ComplexElement<KSTrajIntegratorRK87>("integrator_rk87") +
    KSTrajTrajectoryMagneticBuilder::ComplexElement<KSTrajIntegratorRK8>("integrator_rk8") +
    KSTrajTrajectoryMagneticBuilder::ComplexElement<KSTrajInterpolatorFast>("interpolator_fast") +
    KSTrajTrajectoryMagneticBuilder::ComplexElement<KSTrajInterpolatorHermite>("interpolator_hermite") +
    KSTrajTrajectoryMagneticBuilder::ComplexElement<KSTrajInterpolatorContinuousRungeKutta>("interpolator_crk") +
    KSTrajTrajectoryMagneticBuilder::ComplexElement<KSTrajTermPropagation>("term_propagation") +
    KSTrajTrajectoryMagneticBuilder::ComplexElement<KSTrajControlTime>("control_time") +
    KSTrajTrajectoryMagneticBuilder::ComplexElement<KSTrajControlLength>("control_length") +
    KSTrajTrajectoryMagneticBuilder::ComplexElement<KSTrajControlBChange>("control_B_change") +
    KSTrajTrajectoryMagneticBuilder::Attribute<double>("piecewise_tolerance") +
    KSTrajTrajectoryMagneticBuilder::Attribute<unsigned int>("max_segments");

STATICINT sToolboxKSTrajTrajectoryMagnetic =
    KSRootBuilder::ComplexElement<KSTrajTrajectoryMagnetic>("kstraj_trajectory_magnetic");

}  // namespace katrin
