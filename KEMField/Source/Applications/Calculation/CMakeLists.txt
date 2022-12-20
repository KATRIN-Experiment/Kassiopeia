# CMakeLists for KEMField/Applications/Calculation
# Author: T.J. Corona, D. Hilk

option (KEMField_ENABLE_APP "Build KEMField applications" ON)
if (KEMField_ENABLE_APP)

    set (APPLICATIONS_LIBS
        KEMFieldBindings
    )

    if (KEMField_USE_ROOT)
        list (APPEND APPLICATIONS_LIBS KEMRootPlugin)
    endif (KEMField_USE_ROOT)

    if (KEMField_USE_OPENCL)
        list (APPEND APPLICATIONS_LIBS KEMOpenCLPlugin)
    endif (KEMField_USE_OPENCL)

    if (KEMField_USE_PETSc)
        list (APPEND APPLICATIONS_LIBS KEMPETScPlugin)
    endif (KEMField_USE_PETSc)

    if (KEMField_USE_VTK)
        list (APPEND APPLICATIONS_LIBS KEMVTKPlugin)
    endif (KEMField_USE_VTK)


    set_property(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/ComputeSphericalCapacitor.cc
        APPEND PROPERTY COMPILE_DEFINITIONS DEFAULT_DATA_DIR="${KEMField_DATA_INSTALL_DIR}"
    )

    set_property( SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/ComputeSphericalCapacitor.cc
        APPEND PROPERTY COMPILE_DEFINITIONS DEFAULT_OUTPUT_DIR="${KEMField_OUTPUT_INSTALL_DIR}"
    )

    add_executable (ComputeChargeDensities
      ${CMAKE_CURRENT_SOURCE_DIR}/ComputeChargeDensities.cc)
    target_link_libraries (ComputeChargeDensities ${APPLICATIONS_LIBS} )

    add_executable (ComputeChargeDensitiesFromElcd33File
      ${CMAKE_CURRENT_SOURCE_DIR}/ComputeChargeDensitiesFromElcd33File.cc)
    target_link_libraries (ComputeChargeDensitiesFromElcd33File ${APPLICATIONS_LIBS} )

    add_executable (ComputeSphericalCapacitor
      ${CMAKE_CURRENT_SOURCE_DIR}/ComputeSphericalCapacitor.cc)
    target_link_libraries (ComputeSphericalCapacitor ${APPLICATIONS_LIBS} )

    add_executable (ComputePlateCapacitor
      ${CMAKE_CURRENT_SOURCE_DIR}/ComputePlateCapacitor.cc)
    target_link_libraries (ComputePlateCapacitor ${APPLICATIONS_LIBS} )


    kasper_install_executables (
        ComputeChargeDensities
        ComputeChargeDensitiesFromElcd33File
        ComputeSphericalCapacitor
        ComputePlateCapacitor
    )

    # ROOT

    if (KEMField_USE_ROOT)

        add_executable (AspectRatioFromKbdROOT
            ${CMAKE_CURRENT_SOURCE_DIR}/AspectRatioFromKbdROOT.cc)
        target_link_libraries (AspectRatioFromKbdROOT ${APPLICATIONS_LIBS} )

        add_executable (CompareFieldsAndPotentialsROOT
            ${CMAKE_CURRENT_SOURCE_DIR}/CompareFieldsAndPotentialsROOT.cc)
        target_link_libraries (CompareFieldsAndPotentialsROOT ${APPLICATIONS_LIBS} )

        add_executable (DistanceRatioFromKbdROOT
            ${CMAKE_CURRENT_SOURCE_DIR}/DistanceRatioFromKbdROOT.cc)
        target_link_libraries (DistanceRatioFromKbdROOT ${APPLICATIONS_LIBS} )

        kasper_install_executables (
            AspectRatioFromKbdROOT
            CompareFieldsAndPotentialsROOT
            DistanceRatioFromKbdROOT
        )

    endif (KEMField_USE_ROOT)

    #PCL Stuff

    #cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

    #project(greedy_projection)

    #find_package(PCL 1.2 REQUIRED)

    #include_directories(${PCL_INCLUDE_DIRS})
    #link_directories(${PCL_LIBRARY_DIRS})
    #add_definitions(${PCL_DEFINITIONS})

    #add_executable (greedy_projection ${CMAKE_CURRENT_SOURCE_DIR}/greedy_projection.cpp)
    #target_link_libraries (greedy_projection ${PCL_LIBRARIES})

    #kasper_install_executables (
    #  greedy_projection
    #  )

endif (KEMField_ENABLE_APP)
