# CMakeLists for KEMField/Plugins/VTK
# Author: T.J. Corona

option (@PROJECT_NAME@_USE_VTK "Enables the use of VTK for visualization" OFF)
if (@PROJECT_NAME@_USE_VTK)
  find_package(VTK REQUIRED)
  kasper_external_include_directories(${VTK_INCLUDE_DIRS})
  include(${VTK_USE_FILE}) 
  add_cflag (KEMFIELD_USE_VTK)

  kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

  set (VTKPLUGIN_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KEMVTKViewer.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KEMVTKElectromagnetViewer.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KEMVTKFieldCanvas.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KVTKIterationPlotter.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KVTKResidualGraph.hh
    )

  set (VTKPLUGIN_SOURCEFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KEMVTKViewer.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KEMVTKElectromagnetViewer.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KEMVTKFieldCanvas.cc
    )

  if (VTK_MAJOR_VERSION GREATER 5)
    set_property(
      SOURCE
      ${CMAKE_CURRENT_SOURCE_DIR}/src/KEMVTKViewer.cc
      ${CMAKE_CURRENT_SOURCE_DIR}/src/KEMVTKFieldCanvas.cc
      APPEND PROPERTY COMPILE_DEFINITIONS VTK6
      )
  endif ()

  add_library (KEMVTKPlugin SHARED ${VTKPLUGIN_SOURCEFILES})
  target_link_libraries (KEMVTKPlugin KEMSurfaces KEMVisualization
    ${GSL_LIBRARIES} ${OPENCL_LIBRARIES} ${VTK_LIBRARIES})

  kasper_install_headers (${VTKPLUGIN_HEADERFILES})
  kasper_install_libraries (KEMVTKPlugin)

endif (@PROJECT_NAME@_USE_VTK)
