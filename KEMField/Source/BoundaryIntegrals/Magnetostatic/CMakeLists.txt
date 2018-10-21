# CMakeLists for KEMField/MagnetostaticBoundaryIntegrals
# Author: T.J. Corona

kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set (MAGNETOSTATICBOUNDARYINTEGRALS_HEADERFILES
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KMagnetostaticBoundaryIntegrator.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KMagnetostaticRingIntegrator.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KMagnetostaticLineSegmentIntegrator.hh
  )

set (MAGNETOSTATICBOUNDARYINTEGRALS_SOURCEFILES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KMagnetostaticBoundaryIntegrator.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KMagnetostaticRingIntegrator.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KMagnetostaticLineSegmentIntegrator.cc
  )

##################################################

add_library (KEMMagnetostaticBoundaryIntegrals SHARED ${MAGNETOSTATICBOUNDARYINTEGRALS_SOURCEFILES})
target_link_libraries (KEMMagnetostaticBoundaryIntegrals
KEMMath KEMSurfaces KEMCore ${GSL_LIBRARIES})

kasper_install_headers (${MAGNETOSTATICBOUNDARYINTEGRALS_HEADERFILES})
kasper_install_libraries (KEMMagnetostaticBoundaryIntegrals)

