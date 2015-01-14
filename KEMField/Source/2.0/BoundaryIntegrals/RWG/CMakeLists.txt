# CMakeLists for KEMField/RWGBoundaryIntegrals
# Author: T.J. Corona

#kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set (RWGBOUNDARYINTEGRALS_HEADERFILES
#  ${CMAKE_CURRENT_SOURCE_DIR}/include/KRwgBoundaryIntegral.hh
#  ${CMAKE_CURRENT_SOURCE_DIR}/include/KRwgBoundaryIntegral_ElectrostaticDirichletRectangle.hh
  )

set (RWGBOUNDARYINTEGRALS_SOURCEFILES
#  ${CMAKE_CURRENT_SOURCE_DIR}/src/KRwgBoundaryIntegral_ElectrostaticDirichletRectangle.cc
  )

##################################################

#add_library (RwgBoundaryIntegrals STATIC ${BOUNDARYINTEGRALS_SOURCEFILES})
#target_link_libraries (RwgBoundaryIntegrals Geometry Core ${GSL_LIBRARIES})

#kasper_install_headers (${RWGBOUNDARYINTEGRALS_HEADERFILES})
#kasper_install_libraries (RwgBoundaryIntegrals)

