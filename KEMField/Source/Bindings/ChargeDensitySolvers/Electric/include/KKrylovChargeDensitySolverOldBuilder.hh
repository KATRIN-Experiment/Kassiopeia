/*
 * KKrylovChargeDensitySolverOldBuilder.hh
 *
 *  Created on: 10 Aug 2015
 *      Author: wolfgang
 */

#ifndef KKRYLOVCHARGEDENSITYSOLVEROLDBUILDER_HH_
#define KKRYLOVCHARGEDENSITYSOLVEROLDBUILDER_HH_

#include "KComplexElement.hh"
#include "KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration.hh"
#include "KFMElectrostaticParametersBuilder.hh"
#include "KKrylovChargeDensitySolverOld.hh"

namespace katrin
{


//! the old Krylov solver only allows a single level preconditioning
//! but the new Krylov solver cannot store intermediate results to restart
//! after the process has been terminated externally (time limit on server, blackout,...)
//!
//! The parameters for the fftm_multiplication and
//! the preconditioner_electrostatic_parameters are the same
//! as for <fast_multipole_matrix>, see its documentation.
//! @example
//! <krylov_charge_density_solver_old
//!     solver_name="gmres"
//!     preconditioner="independent_implicit_krylov"
//!     tolerance="1e-8"
//!     max_iterations="500"
//!     iterations_between_restarts="500"
//!     preconditioner_tolerance="1e-2"
//!     max_preconditioner_iterations="5"
//!     preconditioner_degree="2"
//!     intermediate_save_interval="0"
//!     use_display="true"
//!     show_plot="true"
//!     use_timer="true"
//!     time_limit_in_seconds="100"
//!     time_check_interval="1"
//! >
//!     <fftm_multiplication
//!         strategy="balanced"
//!         top_level_divisions="4"
//!         tree_level_divisions="2"
//!         expansion_degree="5"
//!         maximum_tree_depth="2"
//!         region_expansion_factor="1.1"
//!         use_region_size_estimation="true"
//!         world_cube_center_x="0"
//!         world_cube_center_y="0"
//!         world_cube_center_z="0"
//!         world_cube_length="10"
//!         use_caching="false"
//!         verbosity="3"
//!         insertion_ratio="1.33333333333"
//!     />
//!     <preconditioner_electrostatic_parameters
//!         strategy="balanced"
//!         top_level_divisions="4"
//!         tree_level_divisions="2"
//!         expansion_degree="2"
//!         maximum_tree_depth="2"
//!         region_expansion_factor="1.1"
//!         use_region_size_estimation="true"
//!         world_cube_center_x="0"
//!         world_cube_center_y="0"
//!         world_cube_center_z="0"
//!         world_cube_length="10"
//!         use_caching="false"
//!         verbosity="3"
//!         insertion_ratio="1.33333333333"
//!     />
//! </krylov_charge_density_solver_old>
typedef KComplexElement<KEMField::KKrylovChargeDensitySolverOld> KKrylovChargeDensitySolverOldBuilder;

template<> inline bool KKrylovChargeDensitySolverOldBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "solver_name") {
        std::string name = "";
        aContainer->CopyTo(name);
        fObject->GetSolverConfig()->SetSolverName(name);
        return true;
    }
    if (aContainer->GetName() == "preconditioner") {
        std::string name = "";
        aContainer->CopyTo(name);
        fObject->GetSolverConfig()->SetPreconditionerName(name);
        return true;
    }
    if (aContainer->GetName() == "tolerance") {
        double tolerance = 0.0;
        aContainer->CopyTo(tolerance);
        fObject->GetSolverConfig()->SetSolverTolerance(tolerance);
        return true;
    }
    if (aContainer->GetName() == "max_iterations") {
        unsigned int value = 0;
        aContainer->CopyTo(value);
        fObject->GetSolverConfig()->SetMaxSolverIterations(value);
        return true;
    }
    if (aContainer->GetName() == "iterations_between_restarts") {
        unsigned int value = 0;
        aContainer->CopyTo(value);
        fObject->GetSolverConfig()->SetIterationsBetweenRestart(value);
        return true;
    }
    if (aContainer->GetName() == "preconditioner_tolerance") {
        double value = 0.;
        aContainer->CopyTo(value);
        fObject->GetSolverConfig()->SetPreconditionerTolerance(value);
        return true;
    }
    if (aContainer->GetName() == "max_preconditioner_iterations") {
        unsigned int value = 0;
        aContainer->CopyTo(value);
        fObject->GetSolverConfig()->SetMaxPreconditionerIterations(value);
        return true;
    }
    if (aContainer->GetName() == "preconditioner_degree") {
        unsigned int value = 0;
        aContainer->CopyTo(value);
        fObject->GetSolverConfig()->SetPreconditionerDegree(value);
        return true;
    }
    if (aContainer->GetName() == "intermediate_save_interval") {
        unsigned int value = 0;
        aContainer->CopyTo(value);
        fObject->GetSolverConfig()->SetUseCheckpoints(value);
        fObject->GetSolverConfig()->SetCheckpointFrequency(value);
        return true;
    }
    if (aContainer->GetName() == "use_display") {
        bool value = true;
        aContainer->CopyTo(value);
        fObject->GetSolverConfig()->SetUseDisplay(value);
        return true;
    }
    if (aContainer->GetName() == "show_plot") {
        bool value = true;
        aContainer->CopyTo(value);
        fObject->GetSolverConfig()->SetUsePlot(value);
        return true;
    }
    if (aContainer->GetName() == "use_timer") {
        bool value = true;
        aContainer->CopyTo(value);
        fObject->GetSolverConfig()->SetUseTimer(value);
        return true;
    }
    if (aContainer->GetName() == "time_limit_in_seconds") {
        double value = 0.;
        aContainer->CopyTo(value);
        fObject->GetSolverConfig()->SetTimeLimitSeconds(value);
        return true;
    }
    if (aContainer->GetName() == "time_check_interval") {
        unsigned int value = 0;
        aContainer->CopyTo(value);
        fObject->GetSolverConfig()->SetTimeCheckFrequency(value);
        return true;
    }
    return false;
}

template<> inline bool KKrylovChargeDensitySolverOldBuilder::AddElement(KContainer* anElement)
{
    if (anElement->GetName() == "fftm_multiplication") {
        anElement->ReleaseTo(fObject->GetSolverConfig(),
                             &KEMField::KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration::SetFFTMParams);
        return true;
    }
    if (anElement->GetName() == "preconditioner_electrostatic_parameters") {
        anElement->ReleaseTo(
            fObject->GetSolverConfig(),
            &KEMField::KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration::SetPreconditionerFFTMParams);
        return true;
    }
    return false;
}

} /* namespace katrin */

#endif /* KKRYLOVCHARGEDENSITYSOLVEROLDBUILDER_HH_ */
