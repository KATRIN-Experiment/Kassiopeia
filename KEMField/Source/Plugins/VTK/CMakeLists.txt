# CMakeLists for KEMField/Plugins/VTK
# Author: T.J. Corona

if (KEMField_USE_VTK)

    set (VTKPLUGIN_HEADERFILES
        ${CMAKE_CURRENT_SOURCE_DIR}/include/KEMVTKViewer.hh
        ${CMAKE_CURRENT_SOURCE_DIR}/include/KEMVTKElectromagnetViewer.hh
        ${CMAKE_CURRENT_SOURCE_DIR}/include/KEMVTKFieldCanvas.hh
        ${CMAKE_CURRENT_SOURCE_DIR}/include/KVTKIterationPlotter.hh
        ${CMAKE_CURRENT_SOURCE_DIR}/include/KVTKResidualGraph.hh
        ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMVTKElectrostaticTreeViewer.hh
    )

    set (VTKPLUGIN_SOURCEFILES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/KEMVTKViewer.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/KEMVTKElectromagnetViewer.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/KEMVTKFieldCanvas.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMVTKElectrostaticTreeViewer.cc
    )

    if (VTK_MAJOR_VERSION GREATER 5)
        set_property(
          SOURCE
              ${CMAKE_CURRENT_SOURCE_DIR}/src/KEMVTKViewer.cc
              ${CMAKE_CURRENT_SOURCE_DIR}/src/KEMVTKFieldCanvas.cc
              ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMVTKElectrostaticTreeViewer.cc
          APPEND PROPERTY COMPILE_DEFINITIONS VTK6
        )
    endif ()


    add_library (KEMVTKPlugin SHARED
        ${VTKPLUGIN_SOURCEFILES} ${VTKPLUGIN_HEADERFILES})
    target_include_directories(KEMVTKPlugin
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
    target_link_libraries (KEMVTKPlugin
        PUBLIC
            KEMSurfaces
            KEMElectromagnets
            KEMVisualization
            KFMElectrostatics
            KEMLinearAlgebraSolvers
            KommonVtk
    )

    target_compile_definitions(KEMVTKPlugin PUBLIC KEMFIELD_USE_VTK )

    kasper_install_headers (${VTKPLUGIN_HEADERFILES})
    kasper_install_libraries (KEMVTKPlugin)

endif (KEMField_USE_VTK)
