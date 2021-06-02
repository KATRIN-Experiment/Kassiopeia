# CMakeLists for KEMField/LinearAlgebra/Core
# Author: T.J. Corona

set (LINEARALGEBRACORE_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KMatrix.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSimpleMatrix.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSquareMatrix.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSimpleSquareMatrix.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KVector.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSimpleVector.hh
)

add_library(KEMLinearAlgebraCore INTERFACE)
target_include_directories(KEMLinearAlgebraCore
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)

kasper_install_headers (${LINEARALGEBRACORE_HEADERFILES})
kasper_install_libraries (KEMLinearAlgebraCore)
