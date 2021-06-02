# CMakeLists for KEMField/BoundaryIntegrals
# Author: T.J. Corona

set (BOUNDARYINTEGRALS_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KBoundaryIntegralMatrix.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KBoundaryIntegralVector.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KBoundaryIntegralSolutionVector.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KBoundaryMatrixGenerator.hh
)

add_library(KEMBoundaryIntegralsCore INTERFACE)
target_include_directories(KEMBoundaryIntegralsCore
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)

kasper_install_headers (${BOUNDARYINTEGRALS_HEADERFILES})
kasper_install_libraries (KEMBoundaryIntegralsCore)
