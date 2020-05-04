/*
 * KKrylovChargeDensitySolverBuilder.cc
 *
 *  Created on: 17 Aug 2015
 *      Author: wolfgang
 */

#include "KKrylovChargeDensitySolverBuilder.hh"

#include "KElectrostaticBoundaryFieldBuilder.hh"

using namespace KEMField;

namespace katrin
{

template<> KKrylovChargeDensitySolverBuilder::~KComplexElement() {}


STATICINT sKKrylovChargeDensitySolverStructure =
    KKrylovChargeDensitySolverBuilder::Attribute<std::string>("solver_name") +
    KKrylovChargeDensitySolverBuilder::Attribute<double>("tolerance") +
    KKrylovChargeDensitySolverBuilder::Attribute<unsigned int>("max_iterations") +
    KKrylovChargeDensitySolverBuilder::Attribute<unsigned int>("iterations_between_restarts") +
    KKrylovChargeDensitySolverBuilder::Attribute<unsigned int>("intermediate_save_interval") +
    KKrylovChargeDensitySolverBuilder::Attribute<bool>("use_display") +
    KKrylovChargeDensitySolverBuilder::Attribute<bool>("show_plot") +
    KKrylovChargeDensitySolverBuilder::Attribute<bool>("use_timer") +
    KKrylovChargeDensitySolverBuilder::Attribute<double>("time_limit_in_seconds") +
    KKrylovChargeDensitySolverBuilder::Attribute<unsigned int>("time_check_interval");

STATICINT sKElectrostaticBoundaryField =
    KElectrostaticBoundaryFieldBuilder::ComplexElement<KKrylovChargeDensitySolver>("krylov_bem_solver_new") +
    KElectrostaticBoundaryFieldBuilder::ComplexElement<KKrylovChargeDensitySolver>("krylov_charge_density_solver");


} /* namespace katrin */
