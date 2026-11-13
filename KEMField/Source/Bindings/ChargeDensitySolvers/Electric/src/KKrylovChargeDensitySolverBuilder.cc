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

template<> KKrylovChargeDensitySolverBuilder::~KComplexElement() = default;


STATICINT sKKrylovChargeDensitySolverStructure =
    //! specify the solver type by name. gmres and bicgstab are available.
    KKrylovChargeDensitySolverBuilder::Attribute<std::string>("solver_name") +
    //! specify the desired reduction of the start residual to define convergence.
    //! This defaults to 1e-8, meaning that the algorithm will stop
    //! when the norm of the residual has fallen to 1e-8 times its value
    //! in the first iteration (the norm of the vector describing the desired potential distribution).
    KKrylovChargeDensitySolverBuilder::Attribute<double>("tolerance") +
    //! specify the maximum number of iterations after which the solver
    //! will finish regardless of convergence
    KKrylovChargeDensitySolverBuilder::Attribute<unsigned int>("max_iterations") +
    //! specify the maximuum number of iterations before the algorithm will
    //! restart. This means using the last solution but discarding the constructed
    //! Krylov subspace. For more information on the theory of restarted gmres or bicgstab,
    //! google it.
    KKrylovChargeDensitySolverBuilder::Attribute<unsigned int>("iterations_between_restarts") +
    KKrylovChargeDensitySolverBuilder::Attribute<unsigned int>("intermediate_save_interval") +
    //! display progress in the form of iteration number and residual on the console
    KKrylovChargeDensitySolverBuilder::Attribute<bool>("use_display") +
    //! works with vtk only: display the convergence curve in a graph
    KKrylovChargeDensitySolverBuilder::Attribute<bool>("show_plot") +
    KKrylovChargeDensitySolverBuilder::Attribute<bool>("use_timer") +
    KKrylovChargeDensitySolverBuilder::Attribute<double>("time_limit_in_seconds") +
    KKrylovChargeDensitySolverBuilder::Attribute<unsigned int>("time_check_interval");

STATICINT sKElectrostaticBoundaryField =
    KElectrostaticBoundaryFieldBuilder::ComplexElement<KKrylovChargeDensitySolver>("krylov_bem_solver_new") +
    KElectrostaticBoundaryFieldBuilder::ComplexElement<KKrylovChargeDensitySolver>("krylov_charge_density_solver");


} /* namespace katrin */
