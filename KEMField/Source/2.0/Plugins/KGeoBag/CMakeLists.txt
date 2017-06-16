# CMakeLists for KEMField/Plugins/KGeoBag
# T.J. Corona

cmake_dependent_option (@PROJECT_NAME@_USE_KGEOBAG "Use KGeoBag" ON
  "BUILD_KGEOBAG" OFF)
if (NOT BUILD_KGEOBAG)
  set (@PROJECT_NAME@_USE_KGEOBAG OFF)
  set (@PROJECT_NAME@_USE_KGEOBAG ${@PROJECT_NAME@_USE_KGEOBAG} PARENT_SCOPE)
endif(NOT BUILD_KGEOBAG)

if (@PROJECT_NAME@_USE_KGEOBAG)
  kasper_find_module(Kommon)
  kasper_find_module(KGeoBag)
  add_cflag (KEMFIELD_USE_KGEOBAG)

  kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

  set (KGEOBAGPLUGIN_HEADERFILES
  	${CMAKE_CURRENT_SOURCE_DIR}/include/KElectrostaticBoundaryFieldWithKGeoBag.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KEMVectorConverters.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KGBEM.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KGBEMBuilder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KGBEMConverter.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KGElectromagnet.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KGElectromagnetBuilder.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KGElectromagnetConverter.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KStaticElectromagnetFieldWithKGeoBag.hh
    )

  set (KGEOBAGPLUGIN_SOURCEFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KElectrostaticBoundaryFieldWithKGeoBag.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KGBEM.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KGBEMBuilder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KGBEMConverter.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KGElectromagnet.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KGElectromagnetBuilder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KGElectromagnetConverter.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KStaticElectromagnetFieldWithKGeoBag.cc
    )

  add_library (KEMGeoBagPlugin SHARED ${KGEOBAGPLUGIN_SOURCEFILES})
  target_link_libraries (KEMGeoBagPlugin 
  			KEMSurfaces
  			KEMFieldsElectric
  			KEMFieldsMagnetic
  			KEMElectromagnets
  			${GSL_LIBRARIES}
  			${OPENCL_LIBRARIES}
  			${KGeoBag_LIBRARIES}
  )

  kasper_install_headers (${KGEOBAGPLUGIN_HEADERFILES})
  kasper_install_libraries (KEMGeoBagPlugin)

endif (@PROJECT_NAME@_USE_KGEOBAG)
