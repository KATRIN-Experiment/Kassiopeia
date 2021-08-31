# CMakeLists for KEMField/Visualization
# Author: T.J. Corona

set (KEMVISUALIZATION_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KEMFieldCanvas.hh
)

set (KEMVISUALIZATION_SOURCEFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KEMFieldCanvas.cc
)

add_library (KEMVisualization SHARED
    ${KEMVISUALIZATION_SOURCEFILES} ${KEMVISUALIZATION_HEADERFILES})
target_include_directories(KEMVisualization
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_link_libraries (KEMVisualization
    PUBLIC
        KEMCore
)

kasper_install_headers (${KEMVISUALIZATION_HEADERFILES})
kasper_install_libraries (KEMVisualization)
