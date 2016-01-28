# header
set( INTERSECTION_HEADER_FILES
    Include/KGIntersectable.hh
    Include/KGIntersectorInitializer.hh
    Include/KGRotatedSurfaceIntersector.hh
)

# sources
set( INTERSECTION_SOURCE_FILES
    Source/KGIntersectable.cc
    Source/KGIntersectorInitializer.cc
    Source/KGRotatedSurfaceIntersector.cc
)

# internal
kasper_internal_include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/Include
)

# target
add_library( KGeoBagIntersection SHARED
	${INTERSECTION_SOURCE_FILES}
)
target_link_libraries( KGeoBagIntersection
	KGeoBagCore
	KGeoBagShapes
	${Kommon_LIBRARIES}
)

# install
kasper_install_headers( ${INTERSECTION_HEADER_FILES} )
kasper_install_libraries( KGeoBagIntersection )
