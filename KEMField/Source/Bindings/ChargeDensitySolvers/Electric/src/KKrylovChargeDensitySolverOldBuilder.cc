/*
 * KKrylovChargeDensitySolverOldBuilder.cc
 *
 *  Created on: 10 Aug 2015
 *      Author: wolfgang
 */

#include "KKrylovChargeDensitySolverOldBuilder.hh"

#include "KElectrostaticBoundaryFieldBuilder.hh"
#include "KFMElectrostaticParameters.hh"


using namespace KEMField;

namespace katrin
{

template<> KKrylovChargeDensitySolverOldBuilder::~KComplexElement() = default;

STATICINT sKKrylovChargeDensitySolverOldStructure =
    KKrylovChargeDensitySolverOldBuilder::Attribute<std::string>("solver_name") +
    KKrylovChargeDensitySolverOldBuilder::Attribute<std::string>("preconditioner") +
    KKrylovChargeDensitySolverOldBuilder::Attribute<double>("tolerance") +
    KKrylovChargeDensitySolverOldBuilder::Attribute<unsigned int>("max_iterations") +
    KKrylovChargeDensitySolverOldBuilder::Attribute<unsigned int>("iterations_between_restarts") +
    KKrylovChargeDensitySolverOldBuilder::Attribute<double>("preconditioner_tolerance") +
    KKrylovChargeDensitySolverOldBuilder::Attribute<unsigned int>("max_preconditioner_iterations") +
    KKrylovChargeDensitySolverOldBuilder::Attribute<unsigned int>("preconditioner_degree") +
    KKrylovChargeDensitySolverOldBuilder::Attribute<unsigned int>("intermediate_save_interval") +
    KKrylovChargeDensitySolverOldBuilder::Attribute<bool>("use_display") +
    KKrylovChargeDensitySolverOldBuilder::Attribute<bool>("show_plot") +
    KKrylovChargeDensitySolverOldBuilder::Attribute<bool>("use_timer") +
    KKrylovChargeDensitySolverOldBuilder::Attribute<double>("time_limit_in_seconds") +
    KKrylovChargeDensitySolverOldBuilder::Attribute<unsigned int>("time_check_interval") +
    KKrylovChargeDensitySolverOldBuilder::ComplexElement<KFMElectrostaticParameters>("fftm_multiplication") +
    KKrylovChargeDensitySolverOldBuilder::ComplexElement<KFMElectrostaticParameters>(
        "preconditioner_electrostatic_parameters");

STATICINT sKElectrostaticBoundaryField =
    KElectrostaticBoundaryFieldBuilder::ComplexElement<KKrylovChargeDensitySolverOld>("krylov_bem_solver") +
    KElectrostaticBoundaryFieldBuilder::ComplexElement<KKrylovChargeDensitySolverOld>("krylov_bem_solver_old") +
    KElectrostaticBoundaryFieldBuilder::ComplexElement<KKrylovChargeDensitySolverOld>(
        "krylov_charge_density_solver_old");

} /* namespace katrin */
