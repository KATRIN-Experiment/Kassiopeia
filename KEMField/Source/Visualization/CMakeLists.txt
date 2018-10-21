# CMakeLists for KEMField/Visualization
# Author: T.J. Corona

kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set (KEMVISUALIZATION_HEADERFILES
  ${CMAKE_CURRENT_SOURCE_DIR}/include/KEMFieldCanvas.hh
)

set (KEMVISUALIZATION_SOURCEFILES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KEMFieldCanvas.cc
  )

##################################################

add_library (KEMVisualization SHARED ${KEMVISUALIZATION_SOURCEFILES})
target_link_libraries (KEMVisualization KEMCore)

kasper_install_headers (${KEMVISUALIZATION_HEADERFILES})
kasper_install_libraries (KEMVisualization)

