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

# internal
kasper_internal_include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/Include
)

# target
add_library( KGeoBagMetrics SHARED ${METRICS_SOURCE_FILES} ${METRICS_HEADER_FILES} )
target_link_libraries( KGeoBagMetrics
	KGeoBagCore
	KGeoBagShapes
	${Kommon_LIBRARIES}
)

# install
kasper_install_headers( ${METRICS_HEADER_FILES} )
kasper_install_libraries( KGeoBagMetrics )


