/*
 * KElectrostaticZHFieldSolverBuilder.hh
 *
 *  Created on: 23.07.2015
 *      Author: gosda
 */

#ifndef KELECTRICZHFIELDSOLVERBUILDER_HH_
#define KELECTRICZHFIELDSOLVERBUILDER_HH_

#include "KComplexElement.hh"
#include "KElectricZHFieldSolver.hh"
#include "KElectrostaticBoundaryIntegratorAttributeProcessor.hh"

namespace katrin
{

//! uses the zonal harmonic expansion about an axis to compute fields.
//! only valid if the symmetry parameter is set to axial.
//! @example
//! <zonal_harmonic_field_solver
//!     number_of_bifurcations="-1"
//!     convergence_ratio=".99"
//!     convergence_parameter="1.e-15"
//!     proximity_to_sourcepoint="1.e-12"
//!     number_of_central_coefficients="500"
//!     use_fractional_central_sourcepoint_spacing="false"
//!     central_sourcepoint_spacing="2.e-6"
//!     central_sourcepoint_start="-3.3e-3"
//!     central_sourcepoint_end="3.3e-3"
//!     number_of_remote_coefficients="200"
//!     remote_sourcepoint_start="-1.e-1"
//!     remote_sourcepoint_end="1.e-1"
//! />
typedef KComplexElement<KEMField::KElectricZHFieldSolver> KElectricZHFieldSolverBuilder;

template<> inline bool KElectricZHFieldSolverBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "integrator")
        return AddElectrostaticIntegratorPolicy(fObject, aContainer);

    if (aContainer->GetName() == "number_of_bifurcations") {
        aContainer->CopyTo(fObject->GetParameters(), &KEMField::KZonalHarmonicParameters::SetNBifurcations);
        return true;
    }
    if (aContainer->GetName() == "convergence_ratio") {
        aContainer->CopyTo(fObject->GetParameters(), &KEMField::KZonalHarmonicParameters::SetConvergenceRatio);
        return true;
    }
    if (aContainer->GetName() == "proximity_to_sourcepoint") {
        aContainer->CopyTo(fObject->GetParameters(), &KEMField::KZonalHarmonicParameters::SetProximityToSourcePoint);
        return true;
    }
    if (aContainer->GetName() == "convergence_parameter") {
        aContainer->CopyTo(fObject->GetParameters(), &KEMField::KZonalHarmonicParameters::SetConvergenceParameter);
        return true;
    }
    if (aContainer->GetName() == "coaxiality_tolerance") {
        aContainer->CopyTo(fObject->GetParameters(), &KEMField::KZonalHarmonicParameters::SetCoaxialityTolerance);
        return true;
    }
    if (aContainer->GetName() == "number_of_central_coefficients") {
        aContainer->CopyTo(fObject->GetParameters(), &KEMField::KZonalHarmonicParameters::SetNCentralCoefficients);
        return true;
    }
    if (aContainer->GetName() == "use_fractional_central_sourcepoint_spacing") {
        aContainer->CopyTo(fObject->GetParameters(), &KEMField::KZonalHarmonicParameters::SetCentralFractionalSpacing);
        return true;
    }
    if (aContainer->GetName() == "central_sourcepoint_fractional_distance") {
        aContainer->CopyTo(fObject->GetParameters(), &KEMField::KZonalHarmonicParameters::SetCentralFractionalDistance);
        return true;
    }
    if (aContainer->GetName() == "central_sourcepoint_spacing") {
        aContainer->CopyTo(fObject->GetParameters(), &KEMField::KZonalHarmonicParameters::SetCentralDeltaZ);
        return true;
    }
    if (aContainer->GetName() == "central_sourcepoint_start") {
        aContainer->CopyTo(fObject->GetParameters(), &KEMField::KZonalHarmonicParameters::SetCentralZ1);
        return true;
    }
    if (aContainer->GetName() == "central_sourcepoint_end") {
        aContainer->CopyTo(fObject->GetParameters(), &KEMField::KZonalHarmonicParameters::SetCentralZ2);
        return true;
    }
    if (aContainer->GetName() == "number_of_remote_coefficients") {
        aContainer->CopyTo(fObject->GetParameters(), &KEMField::KZonalHarmonicParameters::SetNRemoteCoefficients);
        return true;
    }
    if (aContainer->GetName() == "number_of_remote_sourcepoints") {
        aContainer->CopyTo(fObject->GetParameters(), &KEMField::KZonalHarmonicParameters::SetNRemoteSourcePoints);
        return true;
    }
    if (aContainer->GetName() == "remote_sourcepoint_start") {
        aContainer->CopyTo(fObject->GetParameters(), &KEMField::KZonalHarmonicParameters::SetRemoteZ1);
        return true;
    }
    if (aContainer->GetName() == "remote_sourcepoint_end") {
        aContainer->CopyTo(fObject->GetParameters(), &KEMField::KZonalHarmonicParameters::SetRemoteZ2);
        return true;
    }
    if (aContainer->GetName() == "use_fixed_sourcepoint_range") {
        aContainer->CopyTo(fObject->GetParameters(), &KEMField::KZonalHarmonicParameters::SetUseFixedRange);
        return true;
    }

    return false;
}


} /* namespace katrin */

#endif /* KELECTRICZHFIELDSOLVERBUILDER_HH_ */
