# CMakeLists for KEMField/LinearAlgebra/Solvers
# Author: T.J. Corona

kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set (PRECONDITIONER_HEADERFILES
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KPreconditioner.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KIdentityPreconditioner.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KJacobiPreconditioner.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KRobinHoodPreconditioner.hh
)

kasper_install_headers (${PRECONDITIONER_HEADERFILES})
