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

# target
add_library( KGeoBagIntersection SHARED
    ${INTERSECTION_SOURCE_FILES} ${INTERSECTION_HEADER_FILES}
)
target_include_directories( KGeoBagIntersection
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include> $<INSTALL_INTERFACE:include>
)
target_link_libraries( KGeoBagIntersection
    PUBLIC
        KGeoBagShapes
)

# install
kasper_install_headers( ${INTERSECTION_HEADER_FILES} )
kasper_install_libraries( KGeoBagIntersection )
