# CMakeLists for KEMField/FieldSolvers/Core
# Author: T.J. Corona

set (FIELDSOLVERSCORE_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KIntegratingFieldSolverTemplate.hh
)

add_library(KEMFieldSolverCore INTERFACE)
target_include_directories(KEMFieldSolverCore
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)

kasper_install_headers (${FIELDSOLVERSCORE_HEADERFILES})
kasper_install_libraries (KEMFieldSolverCore)
