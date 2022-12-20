# CMakeLists for KEMField/FieldSolvers/ZonalHarmonic/Solver
# Author: T.J. Corona

set (ZONALHARMONICSOLVER_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KZonalHarmonicComputer.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KZonalHarmonicComputer.icc
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KElectromagnetZonalHarmonicFieldSolver.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KElectrostaticZonalHarmonicFieldSolver.hh
)

set (ZONALHARMONICSOLVER_SOURCEFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KElectromagnetZonalHarmonicFieldSolver.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KElectrostaticZonalHarmonicFieldSolver.cc
)

add_library (KEMZHSolver SHARED
    ${ZONALHARMONICSOLVER_SOURCEFILES} ${ZONALHARMONICSOLVER_HEADERFILES})
target_include_directories(KEMZHSolver
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_link_libraries (KEMZHSolver
    PUBLIC
        KEMSurfaces
        KEMIntegratingSolver
        KEMZHGenerator
)

kasper_install_headers (${ZONALHARMONICSOLVER_HEADERFILES})
kasper_install_libraries (KEMZHSolver)
