# header
set( INTERSECTION_HEADER_FILES
    Include/KGIntersectable.hh
    Include/KGIntersectorInitializer.hh
    Include/KGRotatedSurfaceIntersector.hh

    Include/KG2DArc.hh
    Include/KG2DArea.hh
    Include/KG2DLineSegment.hh
    Include/KG2DPolyLine.hh
    Include/KG2DPolyLineWithArcs.hh
    Include/KG2DPolygon.hh
    Include/KG2DPolygonWithArcs.hh
    Include/KG2DShape.hh
    Include/KGInfinitePlane.hh
    Include/KGVertexSideDescriptor.hh
)

# sources
set( INTERSECTION_SOURCE_FILES
    Source/KGIntersectable.cc
    Source/KGIntersectorInitializer.cc
    Source/KGRotatedSurfaceIntersector.cc

    Source/KG2DArc.cc
    Source/KG2DLineSegment.cc
    Source/KG2DPolyLine.cc
    Source/KG2DPolyLineWithArcs.cc
    Source/KG2DPolygon.cc
    Source/KG2DPolygonWithArcs.cc
    Source/KGInfinitePlane.cc
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


