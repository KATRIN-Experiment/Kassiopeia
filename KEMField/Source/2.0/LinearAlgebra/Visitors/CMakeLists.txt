# CMakeLists for KEMField/LinearAlgebra/Visitors
# Author: T.J. Corona

kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set (LINEARALGEBRAVISITORS_HEADERFILES
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KIterationTracker.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KIterativeStateReader.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KIterativeStateWriter.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KResidualVector.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KIterationTerminator.hh
)

kasper_install_headers (${LINEARALGEBRAVISITORS_HEADERFILES})
