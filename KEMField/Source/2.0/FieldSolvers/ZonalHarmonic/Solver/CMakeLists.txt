# CMakeLists for KEMField/FieldSolvers/ZonalHarmonic/Solver
# Author: T.J. Corona

kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set (ZONALHARMONICSOLVER_HEADERFILES
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KZonalHarmonicComputer.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KZonalHarmonicComputer.icc
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KElectromagnetZonalHarmonicFieldSolver.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KElectrostaticZonalHarmonicFieldSolver.hh
)

set (ZONALHARMONICSOLVER_SOURCEFILES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KElectromagnetZonalHarmonicFieldSolver.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KElectrostaticZonalHarmonicFieldSolver.cc
  )

##################################################

add_library (KEMZHSolver SHARED ${ZONALHARMONICSOLVER_SOURCEFILES})
target_link_libraries (KEMZHSolver KEMCore KEMSurfaces
KEMElectromagnets KEMZHGenerator KEMElectrostaticBoundaryIntegrals 
KEMMath KEMSurfaces KEMCore KEMFieldExceptions ${GSL_LIBRARIES})

kasper_install_headers (${ZONALHARMONICSOLVER_HEADERFILES})
kasper_install_libraries (KEMZHSolver)

