# headers
set( MESH_HEADER_FILES
	Include/KGMeshMessage.hh
	
    Include/KGMeshElement.hh
    Include/KGMeshRectangle.hh
    Include/KGMeshTriangle.hh
    Include/KGMeshWire.hh
    
    Include/KGMesh.hh
    Include/KGMesherBase.hh
    Include/KGMesher.hh
    
    Complex/Include/KGComplexMesher.hh
    Complex/Include/KGBoxMesher.hh
    Complex/Include/KGCylinderMesher.hh
    Complex/Include/KGRotatedSurfaceMesher.hh
    Complex/Include/KGExtrudedSurfaceMesher.hh
    Complex/Include/KGPortHousingSurfaceMesher.hh
    Complex/Include/KGConicSectPortHousingSurfaceMesher.hh
    Complex/Include/KGRodSurfaceMesher.hh
    Complex/Include/KGBeamSurfaceMesher.hh

    Simple/Include/KGSimpleMesher.hh
    Simple/Include/KGFlattenedCircleSurfaceMesher.hh
    Simple/Include/KGFlattenedCircleSurfaceMesher.hh
    Simple/Include/KGFlattenedPolyLoopSurfaceMesher.hh
    Simple/Include/KGRotatedLineSegmentSurfaceMesher.hh
    Simple/Include/KGRotatedArcSegmentSurfaceMesher.hh
    Simple/Include/KGRotatedPolyLineSurfaceMesher.hh
    Simple/Include/KGRotatedCircleSurfaceMesher.hh
    Simple/Include/KGRotatedPolyLoopSurfaceMesher.hh
    Simple/Include/KGShellLineSegmentSurfaceMesher.hh
    Simple/Include/KGShellArcSegmentSurfaceMesher.hh
    Simple/Include/KGShellPolyLineSurfaceMesher.hh
    Simple/Include/KGShellPolyLoopSurfaceMesher.hh
    Simple/Include/KGShellCircleSurfaceMesher.hh
    Simple/Include/KGExtrudedLineSegmentSurfaceMesher.hh
    Simple/Include/KGExtrudedArcSegmentSurfaceMesher.hh
    Simple/Include/KGExtrudedPolyLineSurfaceMesher.hh
    Simple/Include/KGExtrudedCircleSurfaceMesher.hh
    Simple/Include/KGExtrudedPolyLoopSurfaceMesher.hh
    Simple/Include/KGRotatedLineSegmentSpaceMesher.hh
    Simple/Include/KGRotatedArcSegmentSpaceMesher.hh
    Simple/Include/KGRotatedPolyLineSpaceMesher.hh
    Simple/Include/KGRotatedCircleSpaceMesher.hh
    Simple/Include/KGRotatedPolyLoopSpaceMesher.hh
    Simple/Include/KGExtrudedCircleSpaceMesher.hh
    Simple/Include/KGExtrudedPolyLoopSpaceMesher.hh
)

# sources
set( MESH_SOURCE_FILES
	Source/KGMeshMessage.cc
    
    Source/KGMeshElement.cc
    Source/KGMeshRectangle.cc
    Source/KGMeshTriangle.cc
    Source/KGMeshWire.cc
    
    Source/KGMesh.cc
    Source/KGMesherBase.cc
    Source/KGMesher.cc
    
    Complex/Source/KGComplexMesher.cc
    Complex/Source/KGBoxMesher.cc
    Complex/Source/KGCylinderMesher.cc
    Complex/Source/KGRotatedSurfaceMesher.cc
    Complex/Source/KGExtrudedSurfaceMesher.cc
    Complex/Source/KGPortHousingSurfaceMesher.cc
    Complex/Source/KGConicSectPortHousingSurfaceMesher.cc
    Complex/Source/KGRodSurfaceMesher.cc
    Complex/Source/KGBeamSurfaceMesher.cc

    Simple/Source/KGSimpleMesher.cc
    Simple/Source/KGFlattenedCircleSurfaceMesher.cc
    Simple/Source/KGFlattenedCircleSurfaceMesher.cc
    Simple/Source/KGFlattenedPolyLoopSurfaceMesher.cc
    Simple/Source/KGRotatedLineSegmentSurfaceMesher.cc
    Simple/Source/KGRotatedArcSegmentSurfaceMesher.cc
    Simple/Source/KGRotatedPolyLineSurfaceMesher.cc
    Simple/Source/KGRotatedCircleSurfaceMesher.cc
    Simple/Source/KGRotatedPolyLoopSurfaceMesher.cc
    Simple/Source/KGShellLineSegmentSurfaceMesher.cc
    Simple/Source/KGShellArcSegmentSurfaceMesher.cc
    Simple/Source/KGShellPolyLineSurfaceMesher.cc
    Simple/Source/KGShellPolyLoopSurfaceMesher.cc
    Simple/Source/KGShellCircleSurfaceMesher.cc
    Simple/Source/KGExtrudedLineSegmentSurfaceMesher.cc
    Simple/Source/KGExtrudedArcSegmentSurfaceMesher.cc
    Simple/Source/KGExtrudedPolyLineSurfaceMesher.cc
    Simple/Source/KGExtrudedCircleSurfaceMesher.cc
    Simple/Source/KGExtrudedPolyLoopSurfaceMesher.cc
    Simple/Source/KGRotatedLineSegmentSpaceMesher.cc
    Simple/Source/KGRotatedArcSegmentSpaceMesher.cc
    Simple/Source/KGRotatedPolyLineSpaceMesher.cc
    Simple/Source/KGRotatedCircleSpaceMesher.cc
    Simple/Source/KGRotatedPolyLoopSpaceMesher.cc
    Simple/Source/KGExtrudedCircleSpaceMesher.cc
    Simple/Source/KGExtrudedPolyLoopSpaceMesher.cc
)

# internal
kasper_internal_include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/Include
	${CMAKE_CURRENT_SOURCE_DIR}/Complex/Include
	${CMAKE_CURRENT_SOURCE_DIR}/Simple/Include
)


# target
add_library( KGeoBagMesh SHARED ${MESH_SOURCE_FILES} ${MESH_HEADER_FILES} )
target_link_libraries( KGeoBagMesh
	KGeoBagCore
	KGeoBagShapes
	${Kommon_LIBRARIES}
)

# install
kasper_install_headers( ${MESH_HEADER_FILES} )
kasper_install_libraries( KGeoBagMesh )


