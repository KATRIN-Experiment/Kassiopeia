# CMakeLists for KEMField/FieldSolvers/Integrating
# Author: T.J. Corona

kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set (INTEGRATINGFIELDSOLVER_HEADERFILES
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KElectrostaticIntegratingFieldSolver.hh
  )

set (INTEGRATINGFIELDSOLVER_SOURCEFILES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KElectrostaticIntegratingFieldSolver.cc
  )

##################################################

add_library (KEMIntegratingFieldSolver SHARED ${INTEGRATINGFIELDSOLVER_SOURCEFILES})
target_link_libraries (KEMIntegratingFieldSolver KEMMath KEMSurfaces
KEMElectrostaticBoundaryIntegrals KEMCore)

kasper_install_headers (${INTEGRATINGFIELDSOLVER_HEADERFILES})
kasper_install_libraries (KEMIntegratingFieldSolver)

