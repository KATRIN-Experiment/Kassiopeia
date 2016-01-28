# headers
set( RANDOM_HEADER_FILES
    Include/KGRandomMessage.hh
    Include/KGShapeRandom.hh
    Include/KGRotatedSurfaceRandom.hh
    Include/KGRandomPointGenerator.hh
    Include/KGBoxSurfaceRandom.hh
    Include/KGBoxSpaceRandom.hh
    Include/KGDiskSurfaceRandom.hh
    Include/KGCylinderSpaceRandom.hh
    Include/KGCylinderSurfaceRandom.hh
    Include/KGCutConeSpaceRandom.hh
    Include/KGCutConeSurfaceRandom.hh
    Include/KGCutConeTubeSpaceRandom.hh
    Include/KGConeSpaceRandom.hh
    Include/KGConeSurfaceRandom.hh
    Include/KGGenericSpaceRandom.hh
    Include/KGGenericSurfaceRandom.hh
)

# sources
set( RANDOM_SOURCE_FILES
    Source/KGRandomMessage.cc
    Source/KGShapeRandom.cc
    Source/KGRotatedSurfaceRandom.cc
    Source/KGBoxSurfaceRandom.cc
    Source/KGBoxSpaceRandom.cc
    Source/KGDiskSurfaceRandom.cc
    Source/KGCylinderSpaceRandom.cc
    Source/KGCylinderSurfaceRandom.cc
    Source/KGCutConeSpaceRandom.cc
    Source/KGCutConeSurfaceRandom.cc
    Source/KGCutConeTubeSpaceRandom.cc
    Source/KGConeSpaceRandom.cc
    Source/KGConeSurfaceRandom.cc
    Source/KGGenericSpaceRandom.cc
    Source/KGGenericSurfaceRandom.cc
)

# internal
kasper_internal_include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/Include
)

# target
add_library( KGeoBagRandom SHARED ${RANDOM_SOURCE_FILES} ${RANDOM_HEADER_FILES} )
target_link_libraries( KGeoBagRandom
    KGeoBagCore
    KGeoBagShapes
    KGeoBagMetrics
    ${Kommon_LIBRARIES}
)

# install
kasper_install_headers( ${RANDOM_HEADER_FILES} )
kasper_install_libraries( KGeoBagRandom )


