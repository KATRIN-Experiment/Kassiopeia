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
    //! specify the solver type by name. gmres and bicgstab are available.
    KKrylovChargeDensitySolverOldBuilder::Attribute<std::string>("solver_name") +
    //! specify the preconditioner type. Options are:
    //!
    //! - ``none``: use no preconditioner
    //! - ``jacobi``: use the inverse of the diagonal as preconditioner
    //! - ``implicit_krylov``: use an inner Krylov solver with a different FFTM matrix defined by
    //!   the attributes of the Krylov solver.
    //! - ``independent_implicit_krylov``: use an inner Krylov solver with a different FFTM matrix defined by
    //!   the preconditioner_electrostatic_parameters element. This is more
    //!   flexible than the other Krylov preconditioner option.
    KKrylovChargeDensitySolverOldBuilder::Attribute<std::string>("preconditioner") +
    //! specify the desired reduction of the start residual to define convergence.
    //! This defaults to 1e-8, meaning that the algorithm will stop
    //! when the norm of the residual has fallen to 1e-8 times its value
    //! in the first iteration (the norm of the vector describing the desired potential distribution).
    KKrylovChargeDensitySolverOldBuilder::Attribute<double>("tolerance") +
    //! specify the maximum number of iterations after which the solver
    //! will finish regardless of convergence
    KKrylovChargeDensitySolverOldBuilder::Attribute<unsigned int>("max_iterations") +
    //! specify the maximuum number of iterations before the algorithm will
    //! restart. This means using the last solution but discarding the constructed
    //! Krylov subspace. For more information on the theory of restarted gmres or bicgstab,
    //! google it.
    KKrylovChargeDensitySolverOldBuilder::Attribute<unsigned int>("iterations_between_restarts") +
    //! only for the implicit_krylov preconditioner!
    //! define the tolerance just as in the main procedure
    KKrylovChargeDensitySolverOldBuilder::Attribute<double>("preconditioner_tolerance") +
    //! only for the implicit_krylov preconditioner!
    //! define after how many iterations the precondiitoner stops regardless
    //! of convergence
    KKrylovChargeDensitySolverOldBuilder::Attribute<unsigned int>("max_preconditioner_iterations") +
    //! only for the implicit_krylov preconditioner!
    //! the maximum expansion degree of the multipole and local expansions
    //! used in the preconditioner
    KKrylovChargeDensitySolverOldBuilder::Attribute<unsigned int>("preconditioner_degree") +
    //! the number of main iteration steps between subsequent saves of the
    //! intermediate state to the disk
    KKrylovChargeDensitySolverOldBuilder::Attribute<unsigned int>("intermediate_save_interval") +
    //! display progress in the form of iteration number and residual on the console
    KKrylovChargeDensitySolverOldBuilder::Attribute<bool>("use_display") +
    //! works with vtk only: display the convergence curve in a graph
    KKrylovChargeDensitySolverOldBuilder::Attribute<bool>("show_plot") +
    //! boolean option to stop the programm at the end of a main iteration
    //! after a specified amount of time has passed.
    //! For time_check_interval greater than one, the timer will be called
    //! less often to check if the programm should terminate.
    KKrylovChargeDensitySolverOldBuilder::Attribute<bool>("use_timer") +
    //! time after which the timer will terminate when the then current
    //! iteration finishes
    KKrylovChargeDensitySolverOldBuilder::Attribute<double>("time_limit_in_seconds") +
    //! the number of main iterations between calls to the timer enabling
    //! it to stop the process.
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
