# CMakeLists for KEMField/Plugins/VTK
# Author: T.J. Corona

if (@PROJECT_NAME@_USE_VTK)
  add_cflag (KEMFIELD_USE_VTK)

  kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

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
      APPEND PROPERTY COMPILE_DEFINITIONS VTK6
      )
  endif ()
  
  set( VTKPLUGIN_LINK_LIBRARIES
    KEMSurfaces
    KEMVisualization
    KFMCore
    KFMElectrostatics
    ${GSL_LIBRARIES}
    ${OPENCL_LIBRARIES}
    ${VTK_LIBRARIES}
  )  

  add_library (KEMVTKPlugin SHARED ${VTKPLUGIN_SOURCEFILES})
  target_link_libraries (KEMVTKPlugin ${VTKPLUGIN_LINK_LIBRARIES})

  kasper_install_headers (${VTKPLUGIN_HEADERFILES})
  kasper_install_libraries (KEMVTKPlugin)

endif (@PROJECT_NAME@_USE_VTK)
