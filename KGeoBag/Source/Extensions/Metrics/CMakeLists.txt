# headers
set( METRICS_HEADER_FILES
    Include/KGMetrics.hh
    Include/KGMetricsMessage.hh
)

# sources
set( METRICS_SOURCE_FILES
    Source/KGMetrics.cc
    Source/KGMetricsMessage.cc
)

# target
add_library( KGeoBagMetrics SHARED
    ${METRICS_SOURCE_FILES} ${METRICS_HEADER_FILES})
target_include_directories( KGeoBagMetrics
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include> $<INSTALL_INTERFACE:include>)
target_link_libraries( KGeoBagMetrics
    PUBLIC
        KGeoBagShapes
)

# install
kasper_install_headers( ${METRICS_HEADER_FILES} )
kasper_install_libraries( KGeoBagMetrics )
