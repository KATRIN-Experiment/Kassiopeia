# CMakeLists for KEMField/MagnetostaticBoundaryIntegrals
# Author: T.J. Corona

set (MAGNETOSTATICBOUNDARYINTEGRALS_HEADERFILES
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KMagnetostaticBoundaryIntegrator.hh
  #${CMAKE_CURRENT_SOURCE_DIR}/include/KMagnetostaticRingIntegrator.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KMagnetostaticLineSegmentIntegrator.hh
  )

set (MAGNETOSTATICBOUNDARYINTEGRALS_SOURCEFILES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KMagnetostaticBoundaryIntegrator.cc
  #${CMAKE_CURRENT_SOURCE_DIR}/src/KMagnetostaticRingIntegrator.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KMagnetostaticLineSegmentIntegrator.cc
)

##################################################

add_library (KEMMagnetostaticBoundaryIntegrals SHARED
    ${MAGNETOSTATICBOUNDARYINTEGRALS_SOURCEFILES} ${MAGNETOSTATICBOUNDARYINTEGRALS_HEADERFILES})
target_include_directories(KEMMagnetostaticBoundaryIntegrals
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_link_libraries (KEMMagnetostaticBoundaryIntegrals
    PUBLIC
        KEMCore
        KEMSurfaces
        KEMBoundaryIntegralsCore
)

kasper_install_headers (${MAGNETOSTATICBOUNDARYINTEGRALS_HEADERFILES})
kasper_install_libraries (KEMMagnetostaticBoundaryIntegrals)
