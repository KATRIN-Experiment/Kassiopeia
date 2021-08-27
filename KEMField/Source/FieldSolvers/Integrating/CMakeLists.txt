# CMakeLists for KEMField/FieldSolvers/Integrating
# Author: Daniel Hilk

set (INTEGRATINGFIELDSOLVER_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KElectrostaticIntegratingFieldSolver.hh
)

add_library(KEMIntegratingSolver INTERFACE)
target_include_directories(KEMIntegratingSolver
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_link_libraries(KEMIntegratingSolver INTERFACE KEMFieldSolverCore)

kasper_install_headers (${INTEGRATINGFIELDSOLVER_HEADERFILES})
kasper_install_libraries (KEMIntegratingSolver)
