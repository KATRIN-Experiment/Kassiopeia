# CMakeLists for KEMField/ExternalFields/Electromagnets
# Author: T.J. Corona

set (ELECTROMAGNETS_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KCurrentLoop.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KCoil.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KElectromagnet.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSolenoid.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KLineCurrent.hh

    ${CMAKE_CURRENT_SOURCE_DIR}/include/KElectromagnetTypes.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KElectromagnetVisitor.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KElectromagnetContainer.hh

    ${CMAKE_CURRENT_SOURCE_DIR}/include/KCurrentLoopIntegrator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KCoilIntegrator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KSolenoidIntegrator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KLineCurrentIntegrator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KElectromagnetIntegrator.hh

    ${CMAKE_CURRENT_SOURCE_DIR}/include/KElectromagnetIntegratingFieldSolver.hh
)

set (ELECTROMAGNETS_SOURCEFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KCurrentLoop.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KCoil.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KSolenoid.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KLineCurrent.cc

    ${CMAKE_CURRENT_SOURCE_DIR}/src/KCurrentLoopIntegrator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KCoilIntegrator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KSolenoidIntegrator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KLineCurrentIntegrator.cc

    ${CMAKE_CURRENT_SOURCE_DIR}/src/KElectromagnetIntegratingFieldSolver.cc
)

add_library (KEMElectromagnets SHARED
    ${ELECTROMAGNETS_SOURCEFILES} ${ELECTROMAGNETS_HEADERFILES})
target_include_directories(KEMElectromagnets
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_link_libraries (KEMElectromagnets
    PUBLIC
        KEMCore
        KEMMath
        KEMFieldSolverCore
)

kasper_install_headers (${ELECTROMAGNETS_HEADERFILES})
kasper_install_libraries (KEMElectromagnets)
