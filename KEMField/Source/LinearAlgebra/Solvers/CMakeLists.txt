# CMakeLists for KEMField/LinearAlgebra/Solvers
# Author: T.J. Corona

set (LINEARALGEBRASOLVERS_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KIterativeSolver.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KGaussSeidel.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KGaussSeidel_SingleThread.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KGaussianElimination.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KRobinHood.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KRobinHood_SingleThread.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KMultiElementRobinHood.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KMultiElementRobinHood_SingleThread.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSuccessiveSubspaceCorrection.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSuccessiveSubspaceCorrection_SingleThread.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KIterativeKrylovSolver.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KKrylovSolverConfiguration.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KKrylovSolverFactory.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSimpleIterativeKrylovSolver.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KPreconditionedIterativeKrylovSolver.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KGeneralizedMinimalResidual.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KGeneralizedMinimalResidualState.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KPreconditionedGeneralizedMinimalResidual.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KPreconditionedGeneralizedMinimalResidualState.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KBiconjugateGradientStabilized.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KBiconjugateGradientStabilizedState.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KPreconditionedBiconjugateGradientStabilized.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSVDSolver.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KProjectionSolver.hh
)
if (KEMField_USE_MPI)
    list (APPEND LINEARALGEBRASOLVERS_HEADERFILES
        ${CMAKE_CURRENT_SOURCE_DIR}/include/KGaussSeidel_MPI.hh
        ${CMAKE_CURRENT_SOURCE_DIR}/include/KRobinHood_MPI.hh
        ${CMAKE_CURRENT_SOURCE_DIR}/include/KGeneralizedMinimalResidual_MPI.hh
        ${CMAKE_CURRENT_SOURCE_DIR}/include/KPreconditionedGeneralizedMinimalResidual_MPI.hh
)
endif (KEMField_USE_MPI)

set( LINEARALGEBRASOLVERS_SOURCEFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KKrylovSolverConfiguration.cc
)

# library
add_library( KEMLinearAlgebraSolvers SHARED
    ${LINEARALGEBRASOLVERS_SOURCEFILES} ${LINEARALGEBRASOLVERS_HEADERFILES} )
target_include_directories(KEMLinearAlgebraSolvers
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_link_libraries (KEMLinearAlgebraSolvers)
#set_target_properties( KEMLinearAlgebraSolvers PROPERTIES COMPILE_FLAGS -Wno-overloaded-virtual )

# install
kasper_install_headers (${LINEARALGEBRASOLVERS_HEADERFILES})
kasper_install_libraries( KEMLinearAlgebraSolvers )
