# CMakeLists for KEMField/Plugins/Root
# Author: T.J. Corona

if (KEMField_USE_ROOT)

    set (KEMROOT_HEADERFILES
        ${CMAKE_CURRENT_SOURCE_DIR}/include/KEMElectricFieldPointsRootFile.hh
        ${CMAKE_CURRENT_SOURCE_DIR}/include/KEMRootFieldCanvas.hh
        ${CMAKE_CURRENT_SOURCE_DIR}/include/KEMRootSVDSolver.hh
    )

    set (KEMROOT_SOURCEFILES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/KEMElectricFieldPointsRootFile.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/KEMRootFieldCanvas.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/KEMRootSVDSolver.cc
    )

    add_library (KEMRootPlugin SHARED
        ${KEMROOT_SOURCEFILES} ${KEMROOT_HEADERFILES})
    target_include_directories(KEMRootPlugin
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
    target_link_libraries (KEMRootPlugin
        PUBLIC
            KEMVisualization
            KEMMath
            KEMLinearAlgebraCore
            ROOT::Core
            ROOT::Tree
            ROOT::Physics
    )

    target_compile_definitions(KEMRootPlugin PUBLIC KEMFIELD_USE_ROOT )

    kasper_install_headers (${KEMROOT_HEADERFILES})
    kasper_install_libraries (KEMRootPlugin)

endif (KEMField_USE_ROOT)
