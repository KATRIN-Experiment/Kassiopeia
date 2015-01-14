# headers
set( AXIALMESH_HEADER_FILES
	Include/KGAxialMeshMessage.hh
	
    Include/KGAxialMeshElement.hh
    Include/KGAxialMeshLoop.hh
    Include/KGAxialMeshRing.hh
    
    Include/KGAxialMesh.hh
    Include/KGAxialMesherBase.hh
    Include/KGAxialMesher.hh
    
    Simple/Include/KGSimpleAxialMesher.hh
    Simple/Include/KGRotatedLineSegmentSurfaceAxialMesher.hh
	Simple/Include/KGRotatedArcSegmentSurfaceAxialMesher.hh
	Simple/Include/KGRotatedPolyLineSurfaceAxialMesher.hh
	Simple/Include/KGRotatedCircleSurfaceAxialMesher.hh
	Simple/Include/KGRotatedPolyLoopSurfaceAxialMesher.hh
	Simple/Include/KGRotatedLineSegmentSpaceAxialMesher.hh
	Simple/Include/KGRotatedArcSegmentSpaceAxialMesher.hh
	Simple/Include/KGRotatedPolyLineSpaceAxialMesher.hh
	Simple/Include/KGRotatedCircleSpaceAxialMesher.hh
	Simple/Include/KGRotatedPolyLoopSpaceAxialMesher.hh
)

# sources
set( AXIALMESH_SOURCE_FILES
	Source/KGAxialMeshMessage.cc
    
    Source/KGAxialMeshElement.cc
    Source/KGAxialMeshLoop.cc
    Source/KGAxialMeshRing.cc
    
    Source/KGAxialMesh.cc
    Source/KGAxialMesherBase.cc
    Source/KGAxialMesher.cc
    
    Simple/Source/KGSimpleAxialMesher.cc
	Simple/Source/KGRotatedLineSegmentSurfaceAxialMesher.cc
	Simple/Source/KGRotatedArcSegmentSurfaceAxialMesher.cc
	Simple/Source/KGRotatedPolyLineSurfaceAxialMesher.cc
	Simple/Source/KGRotatedCircleSurfaceAxialMesher.cc
	Simple/Source/KGRotatedPolyLoopSurfaceAxialMesher.cc
	Simple/Source/KGRotatedLineSegmentSpaceAxialMesher.cc
	Simple/Source/KGRotatedArcSegmentSpaceAxialMesher.cc
	Simple/Source/KGRotatedPolyLineSpaceAxialMesher.cc
	Simple/Source/KGRotatedCircleSpaceAxialMesher.cc
	Simple/Source/KGRotatedPolyLoopSpaceAxialMesher.cc
)

# internal
kasper_internal_include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/Include
	${CMAKE_CURRENT_SOURCE_DIR}/Simple/Include
)

# target
add_library( KGeoBagAxialMesh SHARED ${AXIALMESH_SOURCE_FILES} ${AXIALMESH_HEADER_FILES} )
target_link_libraries( KGeoBagAxialMesh
    ${Kommon_LIBRARIES}
    KGeoBagCore
    KGeoBagShapes
)

# install
kasper_install_headers( ${AXIALMESH_HEADER_FILES} )
kasper_install_libraries( KGeoBagAxialMesh )


