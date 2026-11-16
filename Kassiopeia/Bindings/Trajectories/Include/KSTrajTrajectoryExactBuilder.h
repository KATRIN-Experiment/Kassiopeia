#ifndef Kassiopeia_KSTrajTrajectoryExactBuilder_h_
#define Kassiopeia_KSTrajTrajectoryExactBuilder_h_

#include "KComplexElement.hh"
#include "KSTrajTrajectoryExact.h"

using namespace Kassiopeia;
namespace katrin
{

//! this trajectory internally uses the exact representation of the particle.
//! in normal fields good results can be obtained only if the control suggests step sizes much smaller than the cyclotron period.
//!
//! Child elements:
//!
//! - integrators: any integrator element (``integrator_*``) may be used.
//! - interpolators: any interpolator element (``interpolator_*``) may be used.
//! - controls: any control element (``control_*``) may be used.
//! - terms: only the propagation and synchrotron terms may be used with this trajectory.
//! @example
//! <kstraj_trajectory_exact name="trajectory_exact">
//!     <integrator_rk8/>
//!     <control_time time="1.e-10"/>
//!     <term_propagation/>
//! </kstraj_trajectory_exact>
typedef KComplexElement<KSTrajTrajectoryExact> KSTrajTrajectoryExactBuilder;

template<> inline bool KSTrajTrajectoryExactBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "piecewise_tolerance") {
        aContainer->CopyTo(fObject, &KSTrajTrajectoryExact::SetPiecewiseTolerance);
        return true;
    }
    if (aContainer->GetName() == "max_segments") {
        aContainer->CopyTo(fObject, &KSTrajTrajectoryExact::SetMaxNumberOfSegments);
        return true;
    }
    if (aContainer->GetName() == "attempt_limit") {
        aContainer->CopyTo(fObject, &KSTrajTrajectoryExact::SetAttemptLimit);
        return true;
    }
    return false;
}

template<> inline bool KSTrajTrajectoryExactBuilder::AddElement(KContainer* aContainer)
{
    if (aContainer->Is<KSTrajExactIntegrator>() == true) {
        aContainer->ReleaseTo(fObject, &KSTrajTrajectoryExact::SetIntegrator);
        return true;
    }
    if (aContainer->Is<KSTrajExactInterpolator>() == true) {
        aContainer->ReleaseTo(fObject, &KSTrajTrajectoryExact::SetInterpolator);
        return true;
    }
    if (aContainer->Is<KSTrajExactDifferentiator>() == true) {
        aContainer->ReleaseTo(fObject, &KSTrajTrajectoryExact::AddTerm);
        return true;
    }
    if (aContainer->Is<KSTrajExactControl>() == true) {
        aContainer->ReleaseTo(fObject, &KSTrajTrajectoryExact::AddControl);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
