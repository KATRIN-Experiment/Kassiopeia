# CMakeLists for FastMultipole/Test
# Author: J. P. Barrett

option (KEMField_ENABLE_FM_APP "Build Fast Multipole applications" OFF)

if (KEMField_ENABLE_FM_APP)

    set (FAST_MULTIPOLE_APP_LIBS
        KEMFieldBindings
    )

    if (KEMField_USE_VTK)
        list (APPEND FAST_MULTIPOLE_APP_LIBS KEMVTKPlugin)
    endif (KEMField_USE_VTK)

    if (KEMField_USE_ROOT)
        list (APPEND FAST_MULTIPOLE_APP_LIBS KEMRootPlugin)
    endif (KEMField_USE_ROOT)

    if (KEMField_USE_OPENCL)
        list (APPEND FAST_MULTIPOLE_APP_LIBS KEMOpenCLPlugin)
    endif (KEMField_USE_OPENCL)

    if (KEMField_USE_PETSc)
        list (APPEND FAST_MULTIPOLE_APP_LIBS KEMPETScPlugin)
    endif (KEMField_USE_PETSc)

    set_property(SOURCE
        #${CMAKE_CURRENT_SOURCE_DIR}/ComputeSphericalCapacitorFastMultipole.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/AccuracyComparisonFastMultipole.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/CompareChargeDensities.cc
        APPEND PROPERTY COMPILE_DEFINITIONS DEFAULT_DATA_DIR="${DATA_INSTALL_DIR}/${KEMField_DATA_DIR}"
    )

    add_executable (CompareChargeDensities
        ${CMAKE_CURRENT_SOURCE_DIR}/CompareChargeDensities.cc)
    target_link_libraries (CompareChargeDensities ${FAST_MULTIPOLE_APP_LIBS})

#    add_executable (ComputeSphericalCapacitorFastMultipole
#        ${CMAKE_CURRENT_SOURCE_DIR}/ComputeSphericalCapacitorFastMultipole.cc)
#    target_link_libraries (ComputeSphericalCapacitorFastMultipole ${FAST_MULTIPOLE_APP_LIBS})

    add_executable (AccuracyComparisonFastMultipole
        ${CMAKE_CURRENT_SOURCE_DIR}/AccuracyComparisonFastMultipole.cc)
    target_link_libraries (AccuracyComparisonFastMultipole ${FAST_MULTIPOLE_APP_LIBS})

    kasper_install_executables (
        CompareChargeDensities
        # ComputeSphericalCapacitorFastMultipole
        AccuracyComparisonFastMultipole
    )

    if (KEMField_USE_ROOT)
        add_executable (ConvertFieldMapToROOT
            ${CMAKE_CURRENT_SOURCE_DIR}/ConvertFieldMapToROOT.cc)
        target_link_libraries (ConvertFieldMapToROOT ${FAST_MULTIPOLE_APP_LIBS})

        kasper_install_executables (
            ConvertFieldMapToROOT
        )

    endif (KEMField_USE_ROOT)

    if (KEMField_USE_VTK)

        add_executable (ConvertFieldMapToVTK
            ${CMAKE_CURRENT_SOURCE_DIR}/ConvertFieldMapToVTK.cc)
        target_link_libraries (ConvertFieldMapToVTK ${FAST_MULTIPOLE_APP_LIBS})

        add_executable (VisualizeElectrostaticMultipoleTree
            ${CMAKE_CURRENT_SOURCE_DIR}/VisualizeElectrostaticMultipoleTree.cc)
        target_link_libraries (VisualizeElectrostaticMultipoleTree ${FAST_MULTIPOLE_APP_LIBS})

        kasper_install_executables (
            ConvertFieldMapToVTK
            VisualizeElectrostaticMultipoleTree
        )

    endif (KEMField_USE_VTK)

    install (DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipoleApplicationFiles
        DESTINATION ${KEMField_CONFIG_INSTALL_DIR}
        PATTERN "*.gitignore" EXCLUDE
    )

endif (KEMField_ENABLE_FM_APP)
