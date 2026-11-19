#ifndef Kassiopeia_KSTrajTrajectoryAdiabaticBuilder_h_
#define Kassiopeia_KSTrajTrajectoryAdiabaticBuilder_h_

#include "KComplexElement.hh"
#include "KSTrajTrajectoryAdiabatic.h"

using namespace Kassiopeia;
namespace katrin
{

//! this trajectory internally uses an adiabatic representation of the particle.
//! this is a good trajectory anywhere the adiabatic approximation is valid.
//!
//! Child elements:
//!
//! - integrators: any integrator element (``integrator_*``) may be used.
//! - interpolators: any interpolator element (``interpolator_*``) may be used.
//! - controls: any control element (``control_*``) may be used.
//! - terms: only the propagation, synchrotron, drift and gyration terms may be used with this trajectory.
//! @example
//! <kstraj_trajectory_adiabatic name="trajectory_adiabatic">
//!     <integrator_rk8/>
//!     <control_time time="1.e-10"/>
//!     <term_propagation/>
//! </kstraj_trajectory_adiabatic>
typedef KComplexElement<KSTrajTrajectoryAdiabatic> KSTrajTrajectoryAdiabaticBuilder;

template<> inline bool KSTrajTrajectoryAdiabaticBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "piecewise_tolerance") {
        aContainer->CopyTo(fObject, &KSTrajTrajectoryAdiabatic::SetPiecewiseTolerance);
        return true;
    }
    if (aContainer->GetName() == "max_segments") {
        aContainer->CopyTo(fObject, &KSTrajTrajectoryAdiabatic::SetMaxNumberOfSegments);
        return true;
    }
    if (aContainer->GetName() == "use_true_position") {
        aContainer->CopyTo(fObject, &KSTrajTrajectoryAdiabatic::SetUseTruePosition);
        return true;
    }
    if (aContainer->GetName() == "cyclotron_fraction") {
        aContainer->CopyTo(fObject, &KSTrajTrajectoryAdiabatic::SetCyclotronFraction);
        return true;
    }
    if (aContainer->GetName() == "attempt_limit") {
        aContainer->CopyTo(fObject, &KSTrajTrajectoryAdiabatic::SetAttemptLimit);
        return true;
    }
    return false;
}

template<> inline bool KSTrajTrajectoryAdiabaticBuilder::AddElement(KContainer* aContainer)
{
    if (aContainer->Is<KSTrajAdiabaticIntegrator>() == true) {
        aContainer->ReleaseTo(fObject, &KSTrajTrajectoryAdiabatic::SetIntegrator);
        return true;
    }
    if (aContainer->Is<KSTrajAdiabaticInterpolator>() == true) {
        aContainer->ReleaseTo(fObject, &KSTrajTrajectoryAdiabatic::SetInterpolator);
        return true;
    }
    if (aContainer->Is<KSTrajAdiabaticDifferentiator>() == true) {
        aContainer->ReleaseTo(fObject, &KSTrajTrajectoryAdiabatic::AddTerm);
        return true;
    }
    if (aContainer->Is<KSTrajAdiabaticControl>() == true) {
        aContainer->ReleaseTo(fObject, &KSTrajTrajectoryAdiabatic::AddControl);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif
