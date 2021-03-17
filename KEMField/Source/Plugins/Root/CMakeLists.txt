# CMakeLists for KEMField/Plugins/Root
# Author: T.J. Corona

if (${PROJECT_NAME}_USE_ROOT)
  kasper_external_include_directories (${ROOT_INCLUDE_DIRS})
  add_cflag (KEMFIELD_USE_ROOT)

  kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

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

  add_library (KEMRootPlugin SHARED ${KEMROOT_SOURCEFILES} ${KEMROOT_HEADERFILES})
  target_link_libraries (KEMRootPlugin KEMVisualization
    KEMMath KEMCore ${Kommon_LIBRARIES} ${KGeoBag_LIBRARIES} ${GSL_LIBRARIES} ${ROOT_LIBRARIES} )

  kasper_install_headers (${KEMROOT_HEADERFILES})
  kasper_install_libraries (KEMRootPlugin)

endif (${PROJECT_NAME}_USE_ROOT)
