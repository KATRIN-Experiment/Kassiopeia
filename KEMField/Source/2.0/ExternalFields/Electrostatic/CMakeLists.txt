# CMakeLists for KEMField/ExternalFields/Electrostatic
# Author: T.J. Corona

kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set (EXTERNALELECTROSTATIC_HEADERFILES
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KElectrostaticConstantFieldSolver.hh
)

kasper_install_headers (${EXTERNALELECTROSTATIC_HEADERFILES})

