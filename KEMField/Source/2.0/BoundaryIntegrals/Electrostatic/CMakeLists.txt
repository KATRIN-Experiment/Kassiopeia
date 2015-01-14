# CMakeLists for KEMField/BoundaryIntegrals/Electrostatic
# Author: T.J. Corona

kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set (ELECTROSTATICBOUNDARYINTEGRALS_HEADERFILES
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KElectrostaticBoundaryIntegrator.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KElectrostaticTriangleIntegrator.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KElectrostaticRectangleIntegrator.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KElectrostaticRingIntegrator.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KElectrostaticLineSegmentIntegrator.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KElectrostaticConicSectionIntegrator.hh
  )

set (ELECTROSTATICBOUNDARYINTEGRALS_SOURCEFILES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KElectrostaticBoundaryIntegrator.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KElectrostaticTriangleIntegrator.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KElectrostaticRectangleIntegrator.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KElectrostaticRingIntegrator.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KElectrostaticLineSegmentIntegrator.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KElectrostaticConicSectionIntegrator.cc
  )

##################################################

add_library (KEMElectrostaticBoundaryIntegrals SHARED ${ELECTROSTATICBOUNDARYINTEGRALS_SOURCEFILES})
target_link_libraries (KEMElectrostaticBoundaryIntegrals
KEMMath KEMSurfaces KEMCore ${GSL_LIBRARIES})

kasper_install_headers (${ELECTROSTATICBOUNDARYINTEGRALS_HEADERFILES})
kasper_install_libraries (KEMElectrostaticBoundaryIntegrals)

