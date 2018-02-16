# CMakeLists for KEMField/FieldSolvers/Core
# Author: T.J. Corona

kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set (FIELDSOLVERSCORE_HEADERFILES
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KIntegratingFieldSolverTemplate.hh
)

kasper_install_headers (${FIELDSOLVERSCORE_HEADERFILES})

