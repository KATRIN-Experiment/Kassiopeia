#ifndef Kassiopeia_KSTrajTrajectoryMagneticBuilder_h_
#define Kassiopeia_KSTrajTrajectoryMagneticBuilder_h_

#include "KComplexElement.hh"
#include "KSTrajTrajectoryMagnetic.h"

using namespace Kassiopeia;
namespace katrin
{

//! this trajectory internally uses a magnetic representation of the particle.
//! this trajectory is used to compute magnetic field lines.
//!
//! Child elements:
//!
//! - integrators: any integrator element (``integrator_*``) may be used.
//! - interpolators: any interpolator element (``interpolator_*``) may be used.
//! - controls: any control element (``control_*``) may be used.
//! - terms: only the propagation term may be used with this trajectory.
//! @example
//! <kstraj_trajectory_magnetic name="trajectory_magnetic">
//!     <integrator_rk8/>
//!     <control_time time="1.e-10"/>
//!     <term_propagation/>
//! </kstraj_trajectory_magnetic>
typedef KComplexElement<KSTrajTrajectoryMagnetic> KSTrajTrajectoryMagneticBuilder;

template<> inline bool KSTrajTrajectoryMagneticBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "piecewise_tolerance") {
        aContainer->CopyTo(fObject, &KSTrajTrajectoryMagnetic::SetPiecewiseTolerance);
        return true;
    }
    if (aContainer->GetName() == "max_segments") {
        aContainer->CopyTo(fObject, &KSTrajTrajectoryMagnetic::SetMaxNumberOfSegments);
        return true;
    }
    if (aContainer->GetName() == "attempt_limit") {
        aContainer->CopyTo(fObject, &KSTrajTrajectoryMagnetic::SetAttemptLimit);
        return true;
    }
    return false;
}

template<> inline bool KSTrajTrajectoryMagneticBuilder::AddElement(KContainer* aContainer)
{
    if (aContainer->Is<KSTrajMagneticIntegrator>() == true) {
        aContainer->ReleaseTo(fObject, &KSTrajTrajectoryMagnetic::SetIntegrator);
        return true;
    }
    if (aContainer->Is<KSTrajMagneticInterpolator>() == true) {
        aContainer->ReleaseTo(fObject, &KSTrajTrajectoryMagnetic::SetInterpolator);
        return true;
    }
    if (aContainer->Is<KSTrajMagneticDifferentiator>() == true) {
        aContainer->ReleaseTo(fObject, &KSTrajTrajectoryMagnetic::AddTerm);
        return true;
    }
    if (aContainer->Is<KSTrajMagneticControl>() == true) {
        aContainer->ReleaseTo(fObject, &KSTrajTrajectoryMagnetic::AddControl);
        return true;
    }
    return false;
}

}  // namespace katrin


#endif
