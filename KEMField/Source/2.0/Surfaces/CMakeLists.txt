# CMakeLists for KEMField/Surfaces
# Author: T.J. Corona

kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set (SURFACES_HEADERFILES
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KBasis.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KBoundary.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KConicSection.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KElectromagneticBasis.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KElectrostaticBasis.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KLineSegment.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KMagnetostaticBasis.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KRectangle.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KRing.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KShape.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KSortedSurfaceContainer.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KSurface.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KSurfaceAction.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KSurfaceContainer.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KSurfaceID.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KSurfacePrimitive.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KSurfaceTypes.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KSurfaceVisitors.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KSymmetryGroup.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KTriangle.hh
#  ${CMAKE_CURRENT_SOURCE_DIR}/include/KOrderedSurfaceContainer.hh
#  ${CMAKE_CURRENT_SOURCE_DIR}/include/KSurfaceOrderingPredicate.hh
)

set (SURFACES_SOURCEFILES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KConicSection.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KLineSegment.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KRectangle.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KRing.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KSortedSurfaceContainer.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KSurface.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KSurfaceContainer.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KTriangle.cc
  )

##################################################

add_library (KEMSurfaces SHARED ${SURFACES_SOURCEFILES})
target_link_libraries (KEMSurfaces KEMCore KEMMath KEMIO)

kasper_install_headers (${SURFACES_HEADERFILES})
kasper_install_libraries (KEMSurfaces)
