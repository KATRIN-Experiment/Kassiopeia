# CMakeLists for KEMField/LinearAlgebra/Core
# Author: T.J. Corona

kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set (LINEARALGEBRACORE_HEADERFILES
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KMatrix.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KSimpleMatrix.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KSquareMatrix.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KSimpleSquareMatrix.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KVector.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KSimpleVector.hh
)

kasper_install_headers (${LINEARALGEBRACORE_HEADERFILES})
