# CMakeLists for KEMField/Interface/KGeoBag
# T.J. Corona, D. Hilk

  kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

  set (KGEOBAGINTERFACE_HEADERFILES
  	${CMAKE_CURRENT_SOURCE_DIR}/include/KGElectrostaticBoundaryField.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KGBEM.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KGBEMConverter.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KGElectromagnet.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KGElectromagnetConverter.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KGStaticElectromagnetField.hh
    )

  set (KGEOBAGINTERFACE_SOURCEFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KGElectrostaticBoundaryField.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KGBEM.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KGBEMConverter.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KGElectromagnet.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KGElectromagnetConverter.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KGStaticElectromagnetField.cc
    )

  add_library (KGeoBagInterface SHARED ${KGEOBAGINTERFACE_SOURCEFILES})
  target_link_libraries (KGeoBagInterface 
  			KEMSurfaces
  			KEMFieldsElectric
  			KEMFieldsMagnetic
  			KEMElectromagnets
  			${GSL_LIBRARIES}
  			${OPENCL_LIBRARIES}
  			${KGeoBag_LIBRARIES}
  )

  kasper_install_headers (${KGEOBAGINTERFACE_HEADERFILES})
  kasper_install_libraries (KGeoBagInterface)
