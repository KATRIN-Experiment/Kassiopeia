/*
 * KKrylovPreconditionerGeneratorBuilder.cc
 *
 *  Created on: 19 Aug 2015
 *      Author: wolfgang
 */

#include "KKrylovPreconditionerGeneratorBuilder.hh"

#include "KKrylovChargeDensitySolverBuilder.hh"

using namespace KEMField;

namespace katrin
{

template<> KKrylovPreconditionerGeneratorBuilder::~KComplexElement() = default;

STATICINT sKKrylovPreconditionerGeneratorStructure =
    //! the name of the particular Krylov solver to be used: GMRES or BiCGStab
    KKrylovPreconditionerGeneratorBuilder::Attribute<std::string>("solver_name") +
    //! the relative reduction of the norm of the residuum to declare the preconditioner converged
    KKrylovPreconditionerGeneratorBuilder::Attribute<double>("tolerance") +
    //! the maximum number of iterations of the preconditioner.
    //! For the preconditioner, this is an important value.
    //! The approximate inverse is best calculated by stopping the iteration
    //! prematurely. By variation of this value, accurracy vs precision can easily be optimized.
    KKrylovPreconditionerGeneratorBuilder::Attribute<unsigned int>("max_iterations") +
    //! same as with the Krylov solver probably never needed in a preconditioner
    //! though because of the low number of iterations
    KKrylovPreconditionerGeneratorBuilder::Attribute<unsigned int>("iterations_between_restarts") +
    KKrylovPreconditionerGeneratorBuilder::Attribute<unsigned int>("intermediate_save_interval") +
    //! see Krylov solver
    KKrylovPreconditionerGeneratorBuilder::Attribute<bool>("use_display") +
    //! see Krylov solver
    KKrylovPreconditionerGeneratorBuilder::Attribute<bool>("show_plot") +
    KKrylovPreconditionerGeneratorBuilder::Attribute<bool>("use_timer") +
    KKrylovPreconditionerGeneratorBuilder::Attribute<double>("time_limit_in_seconds") +
    KKrylovPreconditionerGeneratorBuilder::Attribute<unsigned int>("time_check_interval");

STATICINT sKKrylovChargeDensitySolverStructure =
    KKrylovChargeDensitySolverBuilder::ComplexElement<KKrylovPreconditionerGenerator>("krylov_preconditioner") +
    KKrylovPreconditionerGeneratorBuilder::ComplexElement<KKrylovPreconditionerGenerator>("krylov_preconditioner");


} /* namespace katrin */
