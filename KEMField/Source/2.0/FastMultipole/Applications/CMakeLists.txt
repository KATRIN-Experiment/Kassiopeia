# CMakeLists for FastMultipole/Test
# Author: J. P. Barrett

option (@PROJECT_NAME@_ENABLE_FM_APP "Build fast multipole applications" OFF)
if (@PROJECT_NAME@_ENABLE_FM_APP)
  enable_testing ()
endif (@PROJECT_NAME@_ENABLE_FM_APP)

if (@PROJECT_NAME@_ENABLE_FM_APP)

    kasper_internal_include_directories(${@PROJECT_NAME@_INCLUDE_DIRS})

	add_definitions( -Wno-overloaded-virtual )

    set (FAST_MULTIPOLE_APP_LIBS
        KFMCore
        KFMMath
        KFMMathUtilities
        KVectorMath
        KFMKernel
        KFMElectrostatics
        KFMInterfaceExtraction
        KFMInterfaceFieldSolvers
        KFMUtility
        KEMCore
        KEMFileManipulation
        KEMHashGenerator
        KEMStructuredASCII
        KEMMath
        KEMSurfaces
        KEMIO
        KEMElectrostaticBoundaryIntegrals
        KEMIntegratingFieldSolver
    )

    if (@PROJECT_NAME@_USE_VTK)
    kasper_find_vtk()
    list (APPEND FAST_MULTIPOLE_APP_LIBS
    KEMVTKPlugin
    )
    endif (@PROJECT_NAME@_USE_VTK)

    if (@PROJECT_NAME@_USE_ROOT)
    find_package(ROOT REQUIRED)
    kasper_external_include_directories( ${ROOT_INCLUDE_DIRS} )
    list (APPEND FAST_MULTIPOLE_APP_LIBS
    KEMRootPlugin
    )
    endif (@PROJECT_NAME@_USE_ROOT)

    if (@PROJECT_NAME@_USE_OPENCL)
    kasper_external_include_directories( ${OPENCL_INCLUDE_DIRS} )
    list (APPEND FAST_MULTIPOLE_APP_LIBS
    KEMOpenCLPlugin
    )
    endif (@PROJECT_NAME@_USE_OPENCL)

    if (@PROJECT_NAME@_USE_KGEOBAG)
    kasper_find_module(KGeoBag)
    list (APPEND FAST_MULTIPOLE_APP_LIBS
      KEMGeoBagPlugin
      ${KOMMON_LIBRARIES}
    )
    endif (@PROJECT_NAME@_USE_KGEOBAG)

    if (@PROJECT_NAME@_USE_KMESSAGE)
    kasper_find_module(Kommon)
    list (APPEND FAST_MULTIPOLE_APP_LIBS
      ${KOMMON_LIBRARIES}
    )
    endif (@PROJECT_NAME@_USE_KMESSAGE)

    if (@PROJECT_NAME@_USE_PETSc)
    list (APPEND FAST_MULTIPOLE_APP_LIBS
    KEMPETScPlugin
    )
    endif (@PROJECT_NAME@_USE_PETSc)

    if (@PROJECT_NAME@_USE_VTK)
    list (APPEND FAST_MULTIPOLE_APP_LIBS
      ${VTK_LIBRARIES})
    endif (@PROJECT_NAME@_USE_VTK)

    set_property(
        SOURCE
        #${CMAKE_CURRENT_SOURCE_DIR}/ComputeSphericalCapacitorFastMultipole.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/AccuracyComparisonFastMultipole.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/CompareChargeDensities.cc
        APPEND PROPERTY COMPILE_DEFINITIONS DEFAULT_DATA_DIR="${DATA_INSTALL_DIR}/${@PROJECT_NAME@_DATA_DIR}"
    )

    add_executable (CompareChargeDensities
    ${CMAKE_CURRENT_SOURCE_DIR}/CompareChargeDensities.cc)
    target_link_libraries (CompareChargeDensities ${FAST_MULTIPOLE_APP_LIBS} ${KasperCommon_LIBRARIES})

    kasper_install_executables (
        CompareChargeDensities
    )


    if (KEMField_USE_KGEOBAG)

#        add_executable (ComputeSphericalCapacitorFastMultipole
#        ${CMAKE_CURRENT_SOURCE_DIR}/ComputeSphericalCapacitorFastMultipole.cc)
#        target_link_libraries (ComputeSphericalCapacitorFastMultipole ${FAST_MULTIPOLE_APP_LIBS} ${KasperCommon_LIBRARIES})

        add_executable (AccuracyComparisonFastMultipole
        ${CMAKE_CURRENT_SOURCE_DIR}/AccuracyComparisonFastMultipole.cc)
        target_link_libraries (AccuracyComparisonFastMultipole ${FAST_MULTIPOLE_APP_LIBS} ${KasperCommon_LIBRARIES})

        kasper_install_executables (
#             ComputeSphericalCapacitorFastMultipole
            AccuracyComparisonFastMultipole
        )

    endif (KEMField_USE_KGEOBAG)


    if (KEMField_USE_ROOT)

        add_executable (ConvertFieldMapToROOT
        ${CMAKE_CURRENT_SOURCE_DIR}/ConvertFieldMapToROOT.cc)
        target_link_libraries (ConvertFieldMapToROOT ${FAST_MULTIPOLE_APP_LIBS} ${KasperCommon_LIBRARIES})

        kasper_install_executables (
            ConvertFieldMapToROOT
        )

    endif (KEMField_USE_ROOT)

    if (KEMField_USE_VTK)

        add_executable (ConvertFieldMapToVTK
        ${CMAKE_CURRENT_SOURCE_DIR}/ConvertFieldMapToVTK.cc)
        target_link_libraries (ConvertFieldMapToVTK ${FAST_MULTIPOLE_APP_LIBS} ${KasperCommon_LIBRARIES})

        add_executable (VisualizeElectrostaticMultipoleTree
        ${CMAKE_CURRENT_SOURCE_DIR}/VisualizeElectrostaticMultipoleTree.cc)
        target_link_libraries (VisualizeElectrostaticMultipoleTree ${FAST_MULTIPOLE_APP_LIBS} ${KasperCommon_LIBRARIES})


        kasper_install_executables (
            ConvertFieldMapToVTK
            VisualizeElectrostaticMultipoleTree
        )

    endif (KEMField_USE_VTK)

    install (DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipoleApplicationFiles
      DESTINATION ${@PROJECT_NAME@_CONFIG_INSTALL_DIR}
      PATTERN "*.gitignore" EXCLUDE
      )



endif (@PROJECT_NAME@_ENABLE_FM_APP)
