# CMakeLists for KEMField/FieldSolvers/Integrating
# Author: Daniel Hilk

kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set (INTEGRATINGFIELDSOLVER_HEADERFILES
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KElectrostaticIntegratingFieldSolver.hh
  )

kasper_install_headers (${INTEGRATINGFIELDSOLVER_HEADERFILES})
