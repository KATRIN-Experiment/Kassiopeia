# CMakeLists for KEMField/Applications/Tools
# Author: T.J. Corona, D. Hilk

option (KEMField_ENABLE_APP "Build KEMField applications" ON)
if (KEMField_ENABLE_APP)

    set (APPLICATIONS_LIBS
        KEMFieldBindings
        KGeoBagBindings
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

    add_executable (KEMField
      ${CMAKE_CURRENT_SOURCE_DIR}/KEMField.cc)
    target_link_libraries (KEMField ${APPLICATIONS_LIBS} )

    add_executable (KdbConverter
      ${CMAKE_CURRENT_SOURCE_DIR}/KdbConverter.cc)
    target_link_libraries (KdbConverter ${APPLICATIONS_LIBS} )

    add_executable (TransferEMElement
      ${CMAKE_CURRENT_SOURCE_DIR}/TransferEMElement.cc)
    target_link_libraries (TransferEMElement ${APPLICATIONS_LIBS} )

    add_executable (HashEMGeometry
      ${CMAKE_CURRENT_SOURCE_DIR}/HashEMGeometry.cc)
    target_link_libraries (HashEMGeometry ${APPLICATIONS_LIBS} )

    add_executable (InspectEMFile
      ${CMAKE_CURRENT_SOURCE_DIR}/InspectEMFile.cc)
    target_link_libraries (InspectEMFile ${APPLICATIONS_LIBS} )

    add_executable (WriteKbdToAscii
      ${CMAKE_CURRENT_SOURCE_DIR}/WriteKbdToAscii.cc)
    target_link_libraries (WriteKbdToAscii ${APPLICATIONS_LIBS} )

    add_executable (WriteAsciiToPCD
      ${CMAKE_CURRENT_SOURCE_DIR}/WriteAsciiToPCD.cc)
    target_link_libraries (WriteAsciiToPCD ${APPLICATIONS_LIBS} )

    kasper_install_executables (
        KEMField
        KdbConverter
        TransferEMElement
        HashEMGeometry
        InspectEMFile
        WriteKbdToAscii
        WriteAsciiToPCD
    )

    # VTK

    if (KEMField_USE_VTK)

        add_executable (VisualizeEMGeometry
            ${CMAKE_CURRENT_SOURCE_DIR}/VisualizeEMGeometry.cc)
        target_link_libraries (VisualizeEMGeometry ${APPLICATIONS_LIBS} )

        kasper_install_executables (
            VisualizeEMGeometry
        )

        install (DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ApplicationFiles
            DESTINATION ${KEMField_CONFIG_INSTALL_DIR}
            PATTERN "*.gitignore" EXCLUDE
        )

    endif (KEMField_USE_VTK)

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
