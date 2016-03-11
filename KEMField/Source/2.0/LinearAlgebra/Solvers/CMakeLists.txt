# CMakeLists for KEMField/LinearAlgebra/Solvers
# Author: T.J. Corona

kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

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
  if (@PROJECT_NAME@_USE_MPI)
     list (APPEND LINEARALGEBRASOLVERS_HEADERFILES
          ${CMAKE_CURRENT_SOURCE_DIR}/include/KGaussSeidel_MPI.hh
          ${CMAKE_CURRENT_SOURCE_DIR}/include/KRobinHood_MPI.hh
          ${CMAKE_CURRENT_SOURCE_DIR}/include/KGeneralizedMinimalResidual_MPI.hh
          ${CMAKE_CURRENT_SOURCE_DIR}/include/KPreconditionedGeneralizedMinimalResidual_MPI.hh
    )
  endif (@PROJECT_NAME@_USE_MPI)

kasper_install_headers (${LINEARALGEBRASOLVERS_HEADERFILES})
